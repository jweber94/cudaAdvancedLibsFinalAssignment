#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include "UtilNPP/ImageIO.h"
#include "UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"

class PgmDataGetter {
    public:
        /**
         * @brief checks for all *pgm data (images) in the given folder of the constructor and checks if they are openable
         *        and deliver the possability to get them one after another by path to process them.
         */
        PgmDataGetter(const std::string &pathToFolder) :
            _pathToData{pathToFolder}
        {
            std::cout << "Scanning for data..." << std::endl;
            std::filesystem::directory_iterator dirIt(_pathToData);
            for (auto const &dir_entry : dirIt) {
                if (endsWith(dir_entry.path(), ".pgm")) {
                    _imageNames.push_back(dir_entry.path());
                } else {
                    std::cerr <<"The file " << dir_entry.path() << " is no png image - it will be ignored by the processing" << std::endl;
                }
            }
            _numImages = _imageNames.size();
        }

        /**
         * @brief Get the next image path of the folder that was handed over by the constructur of this instance.
         * 
         * @returns The path to the image so that you can open it up from there. 
         *          If the image is not able to be opened, you will receive "Error".
         *          If you have processed all images from the given path, this method will return an empty string to
         *          indictate that there are no data left to process.  
         */
        std::string getNextImage()
        {
            if (_imageCounter < _numImages)
            {
                auto ret = _imageNames.at(_imageCounter);
                _imageCounter++;

                if (isImageValid(ret))
                    return ret;
                else {
                    return "Error";
                }
            }
            else
            {
                return "";
            }
        }

        /**
         *  @brief Gets the number of images that were found during construction of the instance within the
         *         folder that was handed over to the constructor 
         */
        std::size_t getNumImages() const {
            return _numImages;
        }

        PgmDataGetter() = delete;
        PgmDataGetter(PgmDataGetter &) = default;
        PgmDataGetter& operator = (PgmDataGetter &) = default;
        PgmDataGetter(PgmDataGetter &&) = default;
        PgmDataGetter &operator = (PgmDataGetter &&) = default;

    private:
        bool
        endsWith(const std::string &str, const std::string &suffix)
        {
            return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        bool isImageValid(const std::string& path) {
            int file_errors = 0;
            std::ifstream infile(path.data(), std::ifstream::in);
            if (infile.good())
            {
                file_errors = 0;
                infile.close();
            }
            else
            {
                file_errors++;
                infile.close();
            }
            if (file_errors > 0)
            {
                return false;
            } else {
                return true;
            }
        }

        std::string _pathToData;
        std::vector<std::string> _imageNames;
        std::size_t _numImages;
        std::size_t _imageCounter{0};
};