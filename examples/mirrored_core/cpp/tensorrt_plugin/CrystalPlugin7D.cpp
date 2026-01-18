#include "CrystalPlugin7D.hpp"
#include <iostream>
#include <cstring>
#include <cuda_runtime_api.h>

namespace nvinfer1 {
namespace plugin {

// External CUDA wrapper declaration
extern "C" void launch_project_to_7d_manifold(const float* input, float* output, int n, int dim, float curvature, cudaStream_t stream);

static const char* MANIFOLD_PLUGIN_VERSION{"1"};
static const char* MANIFOLD_PLUGIN_NAME{"CrystalManifoldProjection7D"};

PluginFieldCollection CrystalPlugin7DFactory::mFC{};
std::vector<PluginField> CrystalPlugin7DFactory::mPluginAttributes;

CrystalPlugin7D::CrystalPlugin7D(const std::string name, float curvature)
    : mLayerName(name), mCurvature(curvature) {}

CrystalPlugin7D::CrystalPlugin7D(const std::string name, const void* data, size_t length)
    : mLayerName(name) {
    const char* d = reinterpret_cast<const char*>(data);
    std::memcpy(&mCurvature, d, sizeof(float));
}

const char* CrystalPlugin7D::getPluginType() const noexcept { return MANIFOLD_PLUGIN_NAME; }
const char* CrystalPlugin7D::getPluginVersion() const noexcept { return MANIFOLD_PLUGIN_VERSION; }
int CrystalPlugin7D::getNbOutputs() const noexcept { return 1; }
int CrystalPlugin7D::initialize() noexcept { return 0; }
void CrystalPlugin7D::terminate() noexcept {}
size_t CrystalPlugin7D::getSerializationSize() const noexcept { return sizeof(float); }
void CrystalPlugin7D::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &mCurvature, sizeof(float));
}
void CrystalPlugin7D::destroy() noexcept { delete this; }
void CrystalPlugin7D::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* CrystalPlugin7D::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

DataType CrystalPlugin7D::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept {
    return DataType::kFLOAT;
}

IPluginV2DynamicExt* CrystalPlugin7D::clone() const noexcept {
    return new CrystalPlugin7D(mLayerName, mCurvature);
}

DimsExprs CrystalPlugin7D::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
    return inputs[0];
}

bool CrystalPlugin7D::supportsFormatCombination(int pos, const PluginTensorForm* inOut, int nbInputs, int nbOutputs) noexcept {
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
}

void CrystalPlugin7D::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {}

size_t CrystalPlugin7D::getWorkspaceSize(const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0;
}

int CrystalPlugin7D::enqueue(const IncidentTensorDesc* inputDesc, const IncidentTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    int n = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i) n *= inputDesc[0].dims.d[i];
    int dim = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    if (dim != 7) {
        std::cerr << "CrystalPlugin7D error: input dimension must be 7, found " << dim << std::endl;
        return 1;
    }

    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);

    // Launch configuration
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Direct kernel launch (needs to be linked with the .cu object)
    // Note: In a real plugin, we'd use a wrapper or the raw kernel launch syntax
    // project_to_7d_manifold<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(input, output, n, dim, mCurvature);
    launch_project_to_7d_manifold(input, output, n, dim, mCurvature, stream);

    return 0;
}

// Factory methods
CrystalPlugin7DFactory::CrystalPlugin7DFactory() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("curvature", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CrystalPlugin7DFactory::getPluginName() const noexcept { return MANIFOLD_PLUGIN_NAME; }
const char* CrystalPlugin7DFactory::getPluginVersion() const noexcept { return MANIFOLD_PLUGIN_VERSION; }
const PluginFieldCollection* CrystalPlugin7DFactory::getFieldNames() noexcept { return &mFC; }

IPluginV2DynamicExt* CrystalPlugin7DFactory::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    float curvature = -1.0f;
    for (int i = 0; i < fc->nbFields; ++i) {
        if (std::strcmp(fc->fields[i].name, "curvature") == 0) {
            curvature = *static_cast<const float*>(fc->fields[i].data);
        }
    }
    return new CrystalPlugin7D(name, curvature);
}

IPluginV2DynamicExt* CrystalPlugin7DFactory::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new CrystalPlugin7D(name, serialData, serialLength);
}

void CrystalPlugin7DFactory::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* CrystalPlugin7DFactory::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(CrystalPlugin7DFactory);

} // namespace plugin
} // namespace nvinfer1
