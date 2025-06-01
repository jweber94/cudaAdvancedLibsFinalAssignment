#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <fstream>
#include <memory>
#include <optional>
#include <filesystem>

class ImagePathGetter
{
    public:
        /**
        * @brief checks for all *pgm data (images) in the given folder of the constructor and checks if they are openable
        *        and deliver the possability to get them one after another by path to process them.
        */
        ImagePathGetter(const std::string &pathToFolder) : _pathToData{pathToFolder}
        {
            std::cout << "Scanning for data..." << std::endl;

            // Die directory_iterator nimmt einen std::filesystem::path.
            // Wenn _pathToData ein std::string ist, wandle es explizit um.
            //std::filesystem::directory_iterator dirIt(std::filesystem::path(_pathToData));

            std::filesystem::directory_iterator dirIt(_pathToData); // <- Das ist das Problem
            for (auto const &dir_entry : dirIt)
            {
                // Konvertiere den Pfad der directory_entry zu std::string für endsWith
                if (endsWith(dir_entry.path().string(), ".jpg"))
                {
                    std::cout << "Data path added: " << dir_entry.path() << std::endl;

                    // Hier ist die Korrektur: current_path ist bereits ein std::filesystem::path Objekt
                    std::filesystem::path current_path = dir_entry.path();

                    std::string filenameP = current_path.stem().string();         // Korrekt!
                    std::string directoryP = current_path.parent_path().string(); // Korrekt!
                    std::string extensionP = current_path.extension().string();   // Korrekt!

                    if (!directoryP.empty() && directoryP.back() != std::filesystem::path::preferred_separator)
                    {
                        directoryP += std::filesystem::path::preferred_separator;
                    }

                    size_t underscorePos = filenameP.rfind('_');
                    if (underscorePos == std::string::npos || underscorePos == filenameP.length() - 1)
                    {
                        std::cerr << "ERROR: Invalid path format (no underscore or at end): " << current_path << std::endl;
                        continue; // Diesen Eintrag überspringen
                    }

                    std::string prefix = filenameP.substr(0, underscorePos + 1);  // Z.B. "A02_"
                    std::string number_str = filenameP.substr(underscorePos + 1); // Z.B. "1" oder "2"

                    if (number_str == "1")
                    {
                        // Speichere den kompletten Pfad als std::string
                        _correspondingImages[prefix].first = current_path.string();
                    }
                    else if (number_str == "2")
                    {
                        // Speichere den kompletten Pfad als std::string
                        _correspondingImages[prefix].second = current_path.string();
                    }
                    else
                    {
                        std::cerr << "ERROR: Invalid filename number part (not '1' or '2'): " << current_path << std::endl;
                        continue; // Diesen Eintrag überspringen
                    }
                }
                else
                {
                    std::cerr << "The file " << dir_entry.path() << " is not a .jpg image - it will be ignored by the processing" << std::endl;
                }
            }
            _numImages = _correspondingImages.size();
            _current_iterator = _correspondingImages.begin();
        }

        /**
        * @brief Get the next image path of the folder that was handed over by the constructur of this instance.
        * 
        * @returns The path to the image so that you can open it up from there. 
        *          If the image is not able to be opened, you will receive "Error".
        *          If you have processed all images from the given path, this method will return an empty string to
        *          indictate that there are no data left to process.  
        */
        std::optional<std::pair<std::string, std::pair<std::string, std::string>>> getNextImage()
        {
            if (_current_iterator == _correspondingImages.end())
            {
                return std::nullopt;
            }
            const auto &key_ref = _current_iterator->first;
            const auto &value_ref = _current_iterator->second;

            _current_iterator++;
            return std::make_pair(std::cref(key_ref), std::cref(value_ref));
        }

        /**
        *  @brief Gets the number of images that were found during construction of the instance within the
        *         folder that was handed over to the constructor 
        */
        std::size_t getNumImages() const
        {
            return _numImages;
        }

        ImagePathGetter() = delete;
        ImagePathGetter(ImagePathGetter &) = default;
        ImagePathGetter &operator=(ImagePathGetter &) = default;
        ImagePathGetter(ImagePathGetter &&) = default;
        ImagePathGetter &operator=(ImagePathGetter &&) = default;

    private:
        bool endsWith(const std::string &str, const std::string &suffix)
        {
            return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        bool isImageValid(const std::string &path)
        {
            int file_errors = 0;
            std::ifstream infile(path.data(), std::ifstream::in);
            if (infile.good()) {
                file_errors = 0;
                infile.close();
            } else {
                file_errors++;
                infile.close();
            }
            if (file_errors > 0) {
                return false;
            } else {
                return true;
            }
        }
        
        std::string _pathToData;
        std::unordered_map<std::string, std::pair<std::string, std::string>> _correspondingImages; // name_prefix : <path1, path2>
        std::unordered_map<std::string, std::pair<std::string, std::string>>::const_iterator _current_iterator;
        std::size_t _numImages;
        std::size_t _imageCounter{0};
};