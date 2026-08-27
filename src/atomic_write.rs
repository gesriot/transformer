//! Запись файла целиком — или никак.
//!
//! `File::create` обрезает существующий файл в момент открытия: любая ошибка
//! дальше оставляет на месте прежнего результата обрубок. Здесь запись всегда
//! идёт во временный файл рядом с назначением и заменяет его одним `rename`,
//! поэтому назначение либо старое, либо новое, но никогда не половинчатое.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Счётчик имён временных файлов: два потока не должны выбрать одно имя.
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Удаляет временный файл, если замена не состоялась — в том числе при панике
/// внутри `write`.
struct TempFile {
    path: PathBuf,
    armed: bool,
}

impl TempFile {
    fn keep(mut self) -> PathBuf {
        self.armed = false;
        self.path.clone()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Записать `path` через временный файл рядом с ним.
///
/// `write` получает сам файл и обязан **полностью завершить** запись до
/// возврата: закрыть `ZipWriter::finish`, вызвать `BufWriter::flush`. Буфер,
/// который сбросится позже в `Drop`, проглотит свою ошибку, и тогда «успешная»
/// замена запишет неполные данные.
///
/// Гарантии: после `Ok` в `path` лежит результат `write`; после `Err` прежний
/// файл побитово не изменился (а если его не было, то его и нет), и временных
/// файлов не остаётся.
pub(crate) fn write_atomically<F>(path: &Path, write: F) -> io::Result<()>
where
    F: FnOnce(&mut File) -> io::Result<()>,
{
    let dir = match path.parent() {
        // Пустой родитель у «file.txt» означает текущий каталог, а не корень.
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::other(format!("{}: не путь к файлу", path.display())))?;

    let (temp, mut file) = create_temp(dir, name)?;
    write(&mut file)?;
    // sync_all до замены: иначе rename может стать видимым раньше данных.
    file.sync_all()?;
    // Файл закрывается до rename — иначе замена не пройдёт на Windows.
    drop(file);

    // std::fs::rename заменяет существующее назначение на всех платформах
    // (на Windows — MoveFileEx с MOVEFILE_REPLACE_EXISTING). Удалять
    // назначение заранее нельзя: между удалением и переименованием прежний
    // результат уже потерян.
    let temp_path = temp.keep();
    match fs::rename(&temp_path, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&temp_path);
            Err(e)
        }
    }
}

/// Создать временный файл в каталоге назначения. Соседний каталог — обязателен:
/// `rename` атомарен только внутри одной файловой системы.
fn create_temp(dir: &Path, name: &std::ffi::OsStr) -> io::Result<(TempFile, File)> {
    let pid = std::process::id();
    for _ in 0..32 {
        let seq = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let mut file_name = std::ffi::OsString::from(".");
        file_name.push(name);
        file_name.push(format!(".tmp{pid}-{seq}"));
        let candidate = dir.join(file_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                return Ok((
                    TempFile {
                        path: candidate,
                        armed: true,
                    },
                    file,
                ))
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::other(format!(
        "{}: не удалось создать временный файл",
        dir.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Отдельный каталог на тест: часть проверок смотрит, что в нём НЕТ
    /// лишних файлов.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("transformer_atomic_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn replaces_an_existing_file_and_leaves_no_temporaries() {
        let dir = temp_dir("replace");
        let path = dir.join("data.bin");
        fs::write(&path, b"old").unwrap();

        write_atomically(&path, |f| f.write_all(b"new content")).unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"new content");
        assert_eq!(entries(&dir), vec!["data.bin".to_string()]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failure_after_partial_write_keeps_the_previous_file() {
        let dir = temp_dir("partial");
        let path = dir.join("data.bin");
        fs::write(&path, b"old").unwrap();

        let err = write_atomically(&path, |f| {
            f.write_all(b"half written")?;
            Err(io::Error::other("обрыв на середине"))
        })
        .unwrap_err();

        assert!(err.to_string().contains("обрыв"), "{err}");
        // Прежний файл побитово тот же, временных файлов не осталось.
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(entries(&dir), vec!["data.bin".to_string()]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failure_on_the_first_write_leaves_nothing_behind() {
        let dir = temp_dir("first");
        let path = dir.join("data.bin");

        let err = write_atomically(&path, |f| {
            f.write_all(b"partial")?;
            Err(io::Error::other("нет данных"))
        })
        .unwrap_err();

        assert!(err.to_string().contains("нет данных"), "{err}");
        assert!(!path.exists(), "назначение не должно появиться");
        assert!(entries(&dir).is_empty(), "{:?}", entries(&dir));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn panic_inside_the_writer_also_cleans_up() {
        let dir = temp_dir("panic");
        let path = dir.join("data.bin");
        fs::write(&path, b"old").unwrap();

        let caught = std::panic::catch_unwind(|| {
            let _ = write_atomically(&path, |_| panic!("сбой внутри записи"));
        });

        assert!(caught.is_err(), "паника должна выйти наружу");
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(entries(&dir), vec!["data.bin".to_string()]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn writes_next_to_the_destination_even_without_a_directory_component() {
        // Относительный путь без каталога: временный файл должен лечь в
        // текущий каталог, а не в корень.
        let dir = temp_dir("relative");
        let path = dir.join("plain.txt");
        write_atomically(Path::new(path.to_str().unwrap()), |f| f.write_all(b"x")).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"x");
        fs::remove_dir_all(&dir).unwrap();
    }
}
