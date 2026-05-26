#pragma once

#include <cstddef>
#include <string>

class MappedFile {
  private:
    const std::byte* data_ = nullptr;
    std::size_t      size_ = 0;

  public:
    explicit MappedFile(const std::string& file_path);
    ~MappedFile();

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&)                 = delete;
    MappedFile& operator=(MappedFile&&)      = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }
};
