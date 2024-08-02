#include "ascenddevice.h"

#include "fastllm-acl.h"

#include "executor.h"
#include "utils.h"
#include <acl/acl.h>

namespace fastllm {
    AscendNpuDevice::AscendNpuDevice() {
        this->deviceType = "npu";
        npu::FastllmAclInit();
        this->ops["Linear"] = new AscendLinearOp();
    }

    AscendNpuDevice::~AscendNpuDevice() {
        npu::FastllmAclFinalize();
    }

    bool AscendNpuDevice::Malloc(void **ret, size_t size) {
        *ret = npu::FastllmAclMalloc(size);
        return (*ret != nullptr);
    }

    bool AscendNpuDevice::Free(void *ret) {
        npu::FastllmAclFree(ret);
        return true;
    }

    bool AscendNpuDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        int result = npu::FastllmAclCopyFromHostToDevice(dst, src, size);
        return result == 0;
    }

    bool AscendNpuDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        int result = npu::FastllmAclCopyFromDeviceToHost(dst, src, size);
        return result == 0;
    }

    bool BaseAscendOperator::RunSingleOp(const std::string &opType, const fastllm::DataDict &inputData,
                                         const fastllm::DataDict &outputData, const fastllm::FloatDict &floatParams, 
                                         const fastllm::IntDict &intParams, const std::map <std::string, bool> &boolParams) {
        std::vector<aclTensorDesc *> inputTensors;
        std::vector<aclDataBuffer *> inputBuffers;
        for (auto &pair : inputData)
            npu::FastllmAclToTensor(pair, inputTensors, inputBuffers);
        std::vector<aclTensorDesc *> outputTensors;
        std::vector<aclDataBuffer *> outputBuffers;
        for (auto &pair : outputData)
            npu::FastllmAclToTensor(pair, outputTensors, outputBuffers);
        aclopAttr *attr;
        npu::FastllmAclToOpAttribute(floatParams, intParams, boolParams, &attr);
        bool result = npu::FastllmAclExecute(opType, inputTensors, inputBuffers, 
                               outputTensors, outputBuffers, attr);
        npu::FastllmAclDestoryTensors(inputTensors, inputBuffers, outputTensors, outputBuffers, &attr);
        return result;
    }

    bool BaseAscendOperator::CompileAndRunSingleOp(const std::string &opType, const DataDict &inputData, const fastllm::DataDict &outputData,
                                                   const DynamicShapeDict &dynamicShapes, const FloatDict &floatParams, 
                                                   const IntDict &intParams, const BoolDict &boolParams) {
        std::vector<aclTensorDesc *> inputTensors;
        std::vector<aclDataBuffer *> inputBuffers;
        for (auto &pair : inputData)
            npu::FastllmAclToTensor(pair, inputTensors, inputBuffers);
        std::vector<aclTensorDesc *> outputTensors;
        std::vector<aclDataBuffer *> outputBuffers;
        for (auto &pair : outputData)
            npu::FastllmAclToTensor(pair, outputTensors, outputBuffers);
        aclopAttr *attr;
        npu::FastllmAclToOpAttribute(floatParams, intParams, boolParams, &attr);
        if (warmUpMode) {
            std::vector<aclTensorDesc *> inputTensorsForCompile;
            std::vector<aclTensorDesc *> outputTensorsForCompile;
            for (auto &pair : inputData) {
                auto it = dynamicShapes.find(pair.first);
                if (it != dynamicShapes.end()) {
                    npu::FastllmAclCreateShape(pair, inputTensorsForCompile, it->second.first, it->second.second);
                } else {
                    npu::FastllmAclCreateShape(pair, inputTensorsForCompile);
                }
            }
            for (auto &pair : outputData) {
                auto it = dynamicShapes.find(pair.first);
                if (it != dynamicShapes.end()) {
                    npu::FastllmAclCreateShape(pair, outputTensorsForCompile, it->second.first, it->second.second);
                } else {
                    npu::FastllmAclCreateShape(pair, outputTensorsForCompile);
                }
            }
            npu::FastllmAclInitOp(opType, inputTensorsForCompile, outputTensorsForCompile, attr);
            npu::FastllmAclDestroyShape(inputTensorsForCompile);
            npu::FastllmAclDestroyShape(outputTensorsForCompile);
        }
        bool result = npu::FastllmAclExecuteAfterInit(opType, inputTensors, inputBuffers, 
                                                      outputTensors, outputBuffers, attr);
        npu::FastllmAclDestoryTensors(inputTensors, inputBuffers, outputTensors, outputBuffers, &attr);
        return result;
    }

    bool BaseAscendOperator::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!deviceOk)
            return false;
        Executor *executor = (Executor *) GetExecutor();
        this->warmUpMode = executor->isWarmUpMode();
        return true;
    }

    AscendLinearOp::AscendLinearOp() : 
        BaseAscendOperator("BatchMatMulV2") {}

    void AscendLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
        output.Allocate();
    }

    bool AscendLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!deviceOk)
            return false;
        Executor *executor = (Executor *) GetExecutor();
        this->warmUpMode = executor->isWarmUpMode();

        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        if (input.dataType == DataType::FLOAT16 && weight.dataType == DataType::FLOAT16) {
            return true;
        } else if (input.dataType == DataType::FLOAT32 && weight.dataType == DataType::FLOAT32) {
            return true;
        } else if (input.dataType == DataType::FLOAT32 && weight.dataType == DataType::FLOAT16) {
            return true;
        }
        return false;
    }

    void AscendLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        std::vector<aclTensorDesc *> inputTensors;
        std::vector<aclDataBuffer *> inputBuffers;
        Data fakeReshapedInput;
        fakeReshapedInput.dims = input.dims;
        fakeReshapedInput.strides = input.strides;
        fakeReshapedInput.FakeFrom(input, 0);
        fakeReshapedInput.Reshape({n, m});
        npu::FastllmAclToTensor(std::make_pair("x1", &fakeReshapedInput), inputTensors, inputBuffers);
        npu::FastllmAclToTensor(std::make_pair("x2", &weight), inputTensors, inputBuffers);
        if (bias.dims.size() > 0)
            npu::FastllmAclToTensor(std::make_pair("bias", &bias), inputTensors, inputBuffers);
        std::vector<aclTensorDesc *> outputTensors;
        std::vector<aclDataBuffer *> outputBuffers;
        Data fakeReshapedOutput;
        fakeReshapedOutput.dims = output.dims;
        fakeReshapedOutput.strides = output.strides;
        fakeReshapedOutput.FakeFrom(output, 0);
        fakeReshapedOutput.Reshape({n, k});
        npu::FastllmAclToTensor(std::make_pair("y", &fakeReshapedOutput), outputTensors, outputBuffers);
        aclopAttr *attr;
        npu::FastllmAclToOpAttribute({}, {}, {{"adj_x1", false}, {"adj_x2", true}}, &attr);
        Executor *executor = (Executor *) GetExecutor();
        std::vector<aclTensorDesc *> inputTensorsForCompile;
        std::vector<aclTensorDesc *> outputTensorsForCompile;
        if (warmUpMode) {
            std::vector<std::vector<int64_t>> shapeRanges(1, std::vector<int64_t>({1L, 2048L}));
            npu::FastllmAclCreateShape(std::make_pair("x1", &fakeReshapedInput), inputTensorsForCompile, {0}, shapeRanges);
            npu::FastllmAclCreateShape(std::make_pair("x2", &weight), inputTensorsForCompile);
            if (bias.dims.size() > 0)
                npu::FastllmAclCreateShape(std::make_pair("bias", &bias), inputTensorsForCompile);
            npu::FastllmAclCreateShape(std::make_pair("y", &fakeReshapedOutput), outputTensorsForCompile, {0}, shapeRanges);
        }
        if (input.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT16) {
                if (warmUpMode)
                    deviceOk = npu::FastllmAclInitOp(this->name, inputTensorsForCompile, outputTensorsForCompile, attr);
                deviceOk = npu::FastllmAclExecuteAfterInit(this->name, inputTensors, inputBuffers, 
                                                           outputTensors, outputBuffers, attr);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                if (warmUpMode)
                    deviceOk = npu::FastllmAclInitOp(this->name, inputTensorsForCompile, outputTensorsForCompile, attr);
                deviceOk = npu::FastllmAclExecuteAfterInit(this->name, inputTensors, inputBuffers, 
                                                           outputTensors, outputBuffers, attr);
            } else if (weight.dataType == DataType::FLOAT16) {
                if (warmUpMode)
                    deviceOk = npu::FastllmAclInitOp(this->name, inputTensorsForCompile, outputTensorsForCompile, attr);
                deviceOk = npu::FastllmAclExecuteAfterInit(this->name, inputTensors, inputBuffers, 
                                                           outputTensors, outputBuffers, attr);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport input's dataType.\n");
        }
        npu::FastllmAclDestroyShape(inputTensorsForCompile);
        npu::FastllmAclDestroyShape(outputTensorsForCompile);
        npu::FastllmAclDestoryTensors(inputTensors, inputBuffers, outputTensors, outputBuffers, &attr);
    }

}