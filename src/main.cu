#include <iostream>
#include <string>
#include "CLI11.hpp"

#include "cudaAtScaleFinalAssignment/ImageProcessor.hpp"
#include "cudaAtScaleFinalAssignment/PgmDataGetter.hpp"

int main(int argc, char** argv) {
    // command line parsing
    CLI::App app{"App description"};
    argv = app.ensure_utf8(argv);
    std::string pathToData = "default";
    std::string pathToOutput = "./output";
    app.add_option("-p,--path", pathToData, "Path to the tiff data. We only want the path. The program will iterate over all *.tiff data within this folder (in a non-recursive manner!).");
    app.add_option("-o,--output", pathToOutput, "Output folder where the transformed images should be stored.");
    CLI11_PARSE(app, argc, argv);

    // data input
    PgmDataGetter dataLoader(pathToData);

    // image processor encapsulation
    ImageProcessor imgProcessor(pathToOutput);

    // main loop
    std::cout << "Processing " << dataLoader.getNumImages() << " images" << std::endl;
    bool terminate = false;
    while (!terminate) {
        std::string tmpImgPath = dataLoader.getNextImage();
        if ("" == tmpImgPath || "Error" == tmpImgPath) {
            terminate = true;
            continue;
        }

        if (!imgProcessor.processImage(tmpImgPath)) {
            std::cerr << "Could not process image " << tmpImgPath << " properly." << std::endl;
        }
    }
    std::cout << "All images were processing. Terminating successfully." << std::endl;
    return EXIT_SUCCESS;
}