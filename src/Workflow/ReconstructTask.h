#pragma once
#include "../Estimator/IncrementalMapper.h"
#include "BaseTask.h"
#include <chrono>
#include <map>

enum class ReconstructType { AddModel, AddImage, PrepareRetry, RetryBatch };

struct RegistrationRetryBatch {
	std::vector<image_t> candidates;
	size_t consumed = 0;
	size_t attempted = 0;
	size_t poolSize = 0;
	size_t insufficientSupport = 0;
	size_t cacheMissing = 0;
	size_t succeeded = 0;
	bool failed = false;
	std::map<image_t, std::string> waitingReasons;
};

class ReconstructTask : public BaseTask
{
	Q_OBJECT
	friend struct SfMRegistrationTest;

public:
	explicit ReconstructTask(
		WorkData* data,
		quint32 p_imageID,
		quint32 p_modelID,
		ReconstructType p_type,
		quint32 p_start,
		quint32 p_step,
		QObject* parent = nullptr)
		: BaseTask(data, parent)
		, imageID(p_imageID)
		, modelID(p_modelID)
		, type(p_type)
		, start(p_start)
		, step(p_step) {}
	void run() override;
	void SetRetryBatch(const std::shared_ptr<RegistrationRetryBatch>& batch) { retryBatch = batch; }
	bool Reconstruct(bool init = false);
	bool InitializeEmptyModel(const IncrementalMapper::Options& options, Reconstruction& model, IncrementalMapper& mapper);
	void LocalBundleAdjustment(const IncrementalMapper::Options& options, size_t imageID, IncrementalMapper& mapper);
	void GlobalBundleAdjustment(const IncrementalMapper::Options& options, IncrementalMapper& mapper);
	size_t TriangulateImage(const IncrementalMapper::Options& options, size_t imageID, IncrementalMapper& mapper);
	size_t FilterPoints(const IncrementalMapper::Options& options, IncrementalMapper& mapper);
	size_t FilterImages(const IncrementalMapper::Options& options, IncrementalMapper& mapper);
	void ExtractColors(const IncrementalMapper::Options& options, size_t imageID, Reconstruction& model);
	void CopyEssentialReconstruction();
signals:
	void reconstructTaskFinished(quint32 p_imageID);
	void reconstructionInitial(quint32 p_modelID, bool success);
	void registedInitialImagePair(quint32 imageID1, quint32 imageID2);
	void registedSuccess(quint32 p_imageID);

private:
	quint32 imageID;
	quint32 modelID;
	ReconstructType type;
	quint32 start;
	quint32 step;
	std::shared_ptr<RegistrationRetryBatch> retryBatch;
	bool reconstructionActive = false;
	std::chrono::steady_clock::time_point taskStart;

	size_t GetMatrixValue(image_t i, image_t j);
	std::string RemoveExtension(std::string& filename);  //去掉影像名后缀
	std::unordered_set<image_t> local_ba_images;      //储存本次平差改变的影像
	std::vector<point3D_t> GetImagePoint3D(const std::shared_ptr<Reconstruction> model, const image_t image_id);		//获取当前影像的所有可见3D点
};
