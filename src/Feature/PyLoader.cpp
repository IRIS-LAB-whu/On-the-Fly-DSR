#include "../Base/config.h"
#include "filesystem"
#include "glog/logging.h"
#include "PyLoader.h"
#include <codecvt>
#include <locale>
#include <Windows.h>

using namespace std;
namespace fs = std::filesystem;

namespace {

std::string PyObjectToString(PyObject* object)
{
    if (!object) {
        return "";
    }

    PyObject* stringObject = PyObject_Str(object);
    if (!stringObject) {
        return "";
    }

    const char* value = PyUnicode_AsUTF8(stringObject);
    std::string result = value ? value : "";
    Py_DECREF(stringObject);
    return result;
}

void LogPythonError(const char* context)
{
    if (!PyErr_Occurred()) {
        LOG(ERROR) << context << ": no Python error is set.";
        return;
    }

    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);

    PyObject* tracebackModule = PyImport_ImportModule("traceback");
    if (tracebackModule) {
        PyObject* formatted = PyObject_CallMethod(
            tracebackModule,
            "format_exception",
            "OOO",
            type ? type : Py_None,
            value ? value : Py_None,
            traceback ? traceback : Py_None);

        if (formatted) {
            PyObject* separator = PyUnicode_FromString("");
            PyObject* message = PyUnicode_Join(separator, formatted);
            if (message) {
                LOG(ERROR) << context << ":\n" << PyObjectToString(message);
                Py_DECREF(message);
            }
            Py_XDECREF(separator);
            Py_DECREF(formatted);
        }
        else {
            PyErr_Clear();
            LOG(ERROR) << context << ": " << PyObjectToString(value);
        }

        Py_DECREF(tracebackModule);
    }
    else {
        PyErr_Clear();
        LOG(ERROR) << context << ": " << PyObjectToString(value);
    }

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
}


std::string ResolveSupSceneWeightsPath(const Config& config)
{
    const std::string& configuredPath = config.globalFeature.SupSceneWeightsPath;
    if (!configuredPath.empty() && configuredPath != "auto") {
        return configuredPath;
    }

    fs::path pythonWorkspacePath = fs::absolute(config.python.PythonWorkspace);
    fs::path weightsPath = pythonWorkspacePath / "Supscene" / "checkpoints" / "dinov2_scpp_supscene_1536.pth";
    return weightsPath.string();
}

}

PyObject* PyLoader::DeepImageMatchingInstance = nullptr;
PyObject* PyLoader::GlobalExtractInstance = nullptr;

PyObject* PyLoader::GlobalExtractorFunc = nullptr;
PyObject* PyLoader::LocalExtractorFunc = nullptr;
PyObject* PyLoader::LocalFeatureReadFunc = nullptr;
PyObject* PyLoader::LocalFeatureSaveFunc = nullptr;
PyObject* PyLoader::LocalMatcherFunc = nullptr;

PyThreadState* PyLoader::initThreadState;
std::mutex PyLoader::mtx;

PyLoader::PyLoader()
{
    //初始化Python，设置编译器路径以及环境变量
    LOG(INFO)<< "Initializing PyTorch environment...";
    auto& config = Config::getInstance();
    string pythonHome = config.python.PythonHome;
    string pythonWorkspace = config.python.PythonWorkspace;
    fs::path pythonHomePath = fs::absolute(pythonHome);
    std::wstring wPythonHome = pythonHomePath.wstring();
    SetDllDirectoryW(pythonHomePath.wstring().c_str());     // 设置DLL搜索目录
    fs::path pythonWorkspacePath = fs::absolute(pythonWorkspace);
    fs::path libSitePackages = pythonHomePath / "Lib" / "site-packages";
    fs::path dlls = pythonHomePath / "DLLs";
    fs::path lib = pythonHomePath / "Lib";
    string pythonHomeAbsolute = pythonHomePath.string();
    string pythonPaths =
        libSitePackages.string() + ";" +
        dlls.string() + ";" +
        lib.string() + ";" +
        pythonWorkspacePath.string();
    //wstring wPythonHome(pythonHomeAbsolute.begin() , pythonHomeAbsolute.end());
    wstring wPythonPaths(pythonPaths.begin() , pythonPaths.end());

    PyStatus status;
    PyConfig pyconfig;
    PyConfig_InitPythonConfig(&pyconfig);
    pyconfig.isolated = 1;
    pyconfig.module_search_paths_set = 1;
    PyConfig_SetString(&pyconfig, &pyconfig.home , wPythonHome.c_str());
    PyWideStringList_Append(&pyconfig.module_search_paths, libSitePackages.wstring().c_str());
    PyWideStringList_Append(&pyconfig.module_search_paths, dlls.wstring().c_str());
    PyWideStringList_Append(&pyconfig.module_search_paths, lib.wstring().c_str());
    PyWideStringList_Append(&pyconfig.module_search_paths, pythonWorkspacePath.wstring().c_str());
    //PyConfig_SetString(&pyconfig, &pyconfig.pythonpath_env , wPythonPaths.c_str());
    status = Py_InitializeFromConfig(&pyconfig);
    if (PyStatus_Exception(status))
    {
        LOG(ERROR) << status.err_msg;
        PyConfig_Clear(&pyconfig);
        return;
    }
    PyConfig_Clear(&pyconfig); 

    const char* version = Py_GetVersion();
    LOG(INFO) << "Python version: " + string(version);

    Initialize();

    initThreadState = PyEval_SaveThread();
}

void PyLoader::Initialize()
{
    _import_array();

    auto& config = Config::getInstance();
    string globalFeatureType = config.globalFeature.GlobalFeatureType;

    if (globalFeatureType == "VGG") {
        string moduleName = "GlobalFeatureExtractor";
        string className = "GlobalFeatureExtractor";
        LOG(INFO) << "Loading VGG with NetVLAD global feature extractor...";
        InitializeVGG();
    }
    
    else if (globalFeatureType == "SupScene" || globalFeatureType == "Supscene") {
        LOG(INFO) << "Loading SupScene global feature extractor...";
        InitializeSupScene();
    }
    else {
        // 默认使用 VGG
        string moduleName = "GlobalFeatureExtractor";
        string className = "GlobalFeatureExtractor";
        LOG(WARNING) << "Unknown GlobalFeatureType: " << globalFeatureType << ", defaulting to VGG";
        InitializeVGG();
    } 
}

void PyLoader::InitializeVGG()
{
    PyObject* GlobalExtractorModule = PyImport_ImportModule("GlobalVGG.GlobalFeatureExtractor");
    if (GlobalExtractorModule == NULL) {
        LogPythonError("Failed to import GlobalVGG.GlobalFeatureExtractor");
        LOG(INFO) << "Module GlobalFeatureExtractor not found.";
        return;
    }

    PyObject* globalFeatureExtractorClass = PyObject_GetAttrString(GlobalExtractorModule, "GlobalFeatureExtractor");
    if (!globalFeatureExtractorClass)
    {
        LOG(WARNING) << "Failed to get GlobalFeatureExtractor class!";
        LogPythonError("Failed to get GlobalFeatureExtractor class");
        return;
    }

    PyObject* args = PyTuple_New(4);
    auto& config = Config::getInstance();
    //string modelHome = config.python.GlobalFeatureModel;
    //fs::path modelHomePath = fs::absolute(modelHome);
    string pythonWorkspace = config.python.PythonWorkspace;
    fs::path pythonWorkspacePath = fs::absolute(pythonWorkspace);
    fs::path vggModelPath = pythonWorkspacePath / "GlobalVGG" / "models";
    fs::path vggpth = vggModelPath / "vgg16-397923af.pth";
    fs::path vgg16_64pth = vggModelPath / "VGG16_64_desc_cen.hdf5";
    fs::path vgg_netvladpth = vggModelPath / "VGG16_NetVlad_NoSplit.pth.tar";
    fs::path pca_dimspth = vggModelPath / "PCA_dims32768to2048.model";
    PyTuple_SetItem(args, 0, Py_BuildValue("s", vggpth.string().c_str()));
    PyTuple_SetItem(args, 1, Py_BuildValue("s", vgg16_64pth.string().c_str()));
    PyTuple_SetItem(args, 2, Py_BuildValue("s", vgg_netvladpth.string().c_str()));
    PyTuple_SetItem(args, 3, Py_BuildValue("s", pca_dimspth.string().c_str()));
    PyObject* globalFeatureExtractorClassInstantiation = PyObject_CallObject(globalFeatureExtractorClass, args);
    Py_DECREF(args); Py_DECREF(globalFeatureExtractorClass);
    if (!globalFeatureExtractorClassInstantiation)
    {
        LOG(WARNING) << "Instantiation failed!";
        LogPythonError("Failed to instantiate GlobalFeatureExtractor");
        return;
    }
    GlobalExtractInstance = globalFeatureExtractorClassInstantiation;
    GlobalExtractorFunc = PyObject_GetAttrString(globalFeatureExtractorClassInstantiation, "Extract");
    if (!GlobalExtractorFunc)
    {
        LOG(WARNING) << "Get function failed!";
        LogPythonError("Failed to get GlobalFeatureExtractor.Extract");
        return;
    }
    LOG(INFO) << "The global extractor has been successfully loaded!";

    //Deep Image Matching遗留
    /*PyObject* DIMExtractModule = PyImport_ImportModule("DIMScript");
    if (DIMExtractModule == NULL) {
        PyErr_Print();
        LOG(WARNING) << "Module DIMScript not found.";
        return;
    }
    PyObject* DIMExtractClass = PyObject_GetAttrString(DIMExtractModule , "OTFMatcher");
    if (DIMExtractClass == NULL) {
        PyErr_Print();
        LOG(WARNING) << "Class OTFMatcher not found.";
        return;
    }
    PyObject* DIMExtractClassInstantiation = PyObject_CallObject(DIMExtractClass , nullptr);
    Py_XDECREF(DIMExtractClass);
    if (DIMExtractClassInstantiation == NULL) {
        PyErr_Print();
        LOG(WARNING) << "Instance OTFMatcher create failed.";
        return;
    }
    DeepImageMatchingInstance = DIMExtractClassInstantiation;
    LocalExtractorFunc = PyObject_GetAttrString(DIMExtractClassInstantiation , "extract");
    if (!LocalExtractorFunc)
    {
        LOG(WARNING) << "Get function failed!";
        PyErr_Print();
        return;
    }
    LocalFeatureSaveFunc = PyObject_GetAttrString(DIMExtractClassInstantiation , "save_features");
    if (!LocalFeatureSaveFunc)
    {
        LOG(WARNING) << "Get function failed!";
        PyErr_Print();
        return;
    }
    LOG(INFO) << "The local extractor has been successfully loaded!";

    LocalMatcherFunc = PyObject_GetAttrString(DIMExtractClassInstantiation , "match");
    if (!LocalMatcherFunc)
    {
        LOG(WARNING) << "Get function failed!";
        PyErr_Print();
        return;
    }
    LocalFeatureReadFunc = PyObject_GetAttrString(DIMExtractClassInstantiation , "read_features");
    if (!LocalFeatureReadFunc)
    {
        LOG(WARNING) << "Get function failed!";
        PyErr_Print();
        return;
    }*/
}


void PyLoader::InitializeSupScene()
{
    auto& config = Config::getInstance();

    PyObject* GlobalExtractorModule = PyImport_ImportModule("Supscene.SupSceneFeatureExtractor");
    if (GlobalExtractorModule == NULL) {
        LogPythonError("Failed to import Supscene.SupSceneFeatureExtractor");
        LOG(INFO) << "Module SupSceneFeatureExtractor not found.";
        return;
    }

    PyObject* globalFeatureExtractorClass = PyObject_GetAttrString(GlobalExtractorModule, "SupSceneFeatureExtractor");
    if (!globalFeatureExtractorClass) {
        LOG(ERROR) << "Failed to get SupSceneFeatureExtractor class!";
        LogPythonError("Failed to get SupSceneFeatureExtractor class");
        Py_DECREF(GlobalExtractorModule);
        return;
    }

    const std::string weightsPath = ResolveSupSceneWeightsPath(config);
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, Py_BuildValue("s", weightsPath.c_str()));
    PyTuple_SetItem(args, 1, Py_BuildValue("s", config.globalFeature.SupSceneDevice.c_str()));

    LOG(INFO) << "SupScene config: weights=" << weightsPath
        << ", device=" << config.globalFeature.SupSceneDevice;

    PyObject* globalFeatureExtractorClassInstantiation = PyObject_CallObject(globalFeatureExtractorClass, args);
    Py_DECREF(args);
    Py_DECREF(globalFeatureExtractorClass);
    Py_DECREF(GlobalExtractorModule);

    if (!globalFeatureExtractorClassInstantiation) {
        LOG(ERROR) << "SupSceneFeatureExtractor instantiation failed!";
        LogPythonError("Failed to instantiate SupSceneFeatureExtractor");
        return;
    }

    GlobalExtractInstance = globalFeatureExtractorClassInstantiation;
    GlobalExtractorFunc = PyObject_GetAttrString(globalFeatureExtractorClassInstantiation, "Extract");
    if (!GlobalExtractorFunc) {
        LOG(ERROR) << "Failed to get SupScene Extract function!";
        LogPythonError("Failed to get SupSceneFeatureExtractor.Extract");
        return;
    }

    LOG(INFO) << "SupScene global feature extractor loaded successfully!";
}

PyLoader::~PyLoader() {
    PyEval_RestoreThread(initThreadState);
    Py_Finalize();
}
