//
//  addBizCode.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "addBizCode.hpp"
#include <fstream>
#include <iostream>
#include "logkit.h"
#include "options.hpp"

int addBizCode(const std::string modelFile, const std::string bizCode,
               const common::Options& options, std::unique_ptr<MNN::NetT>& netT) {
    std::ifstream inputFile(modelFile, std::ios::binary);
    inputFile.seekg(0, std::ios::end);
    auto size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    inputFile.read((char*)buffer, size);
    inputFile.close();
    netT = MNN::UnPackNet(buffer);
    CHECK(netT->oplists.size() > 0) << "MNN Molde ERROR: " << modelFile;

    netT->bizCode = bizCode;

    delete[] buffer;

    return 0;
}
