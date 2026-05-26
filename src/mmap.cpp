#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

#include "mmap.h"

MappedFile::MappedFile(const std::string &file_path) {
  // open the file to get the size
  int fd = open(file_path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::system_error(errno, std::system_category(), "open failed: " + file_path);
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    int saved_errno = errno;
    close(fd);
    throw std::system_error(saved_errno, std::system_category(), "fstat failed: " + file_path);
  }

  if (st.st_size <= 0) {
    close(fd);
    throw std::system_error(EINVAL, std::system_category(), "empty or invalid file: " + file_path);
  }

  // mmap!
  // we "map" a st_size bytes of the file into our virtual address space (tracked by kernel via page tables), 
  // that will be used to quickly refer to model weights  (loaded on first forward pass, with page faults) 
  // without extra overhead (no I/O, lazy loading)
  void *p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    int saved_errno = errno;
    close(fd);
    throw std::system_error(saved_errno, std::system_category(), "mmap failed: " + file_path);
  }

  // The mapping keeps its own reference to the file; we no longer need the fd.
  close(fd);

  data_ = reinterpret_cast<const std::byte *>(p);
  size_ = static_cast<std::size_t>(st.st_size);
}

MappedFile::~MappedFile() {
  if (data_ != nullptr) {
    munmap(const_cast<std::byte *>(data_), size_);
  }
}
