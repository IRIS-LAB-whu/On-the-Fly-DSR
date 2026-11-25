#include "../Base/config.h"
#include "../Base/GpuDiagnostics.h"
#include "../Base/misc.h"
#include "../Base/timer.h"
#include "../Estimator/IncrementalMapper.h"
#include "../Scene/Database.h"
#include "../Scene/mesh.h"
#include "glog/logging.h"
#include "ReconstructTask.h"
#include "WorkData.h"
#include <sstream>

#include <boost/format.hpp>

std::unordered_map<image_pair_t, size_t> matchMatrix;
const auto& cfg = Config::getInstance();
const bool Export3DGS = cfg.pipeline.Export3DGS;
bool IsInitialized3DGS = false;

namespace {

size_t AtomicMapValue(
	const tbb::concurrent_map<std::size_t, std::atomic_size_t>& values,
	const image_t imageID)
{
	const auto it = values.find(imageID);
	return it == values.end() ? 0 : it->second.load();
}

void AppendSfmTiming(
	WorkData* data,
	const image_t imageID,
	const std::string& imageName,
	const Reconstruction& model,
	const bool registered,
	const size_t newPoints,
	const size_t filteredPoints,
	const double poseMs,
	const double triangulationMs,
	const double localBAMs)
{
	if (!Config::getInstance().pipeline.OutputTimingFiles) {
		return;
	}
	if (!Database::exportPath.empty()) {
		GpuDiagnostics::SetOutputRoot(
			StringReplace(
				EnsureTrailingSlash(GetParentDir(Database::exportPath)),
				"\\",
				"/") + "DenseReconstruction");
	}

	const size_t globalFeatureMs =
		AtomicMapValue(data->globalFeatureExtractionTimes, imageID);
	const size_t retrievalMs =
		AtomicMapValue(data->globalFeatureRetrievalTimes, imageID);
	const size_t siftMs =
		AtomicMapValue(data->SIFTExtractionTimes, imageID);
	const size_t matchingMs =
		AtomicMapValue(data->SIFTMatchingTimes, imageID);
	const double sfmTotalMs =
		static_cast<double>(globalFeatureMs + retrievalMs + siftMs + matchingMs) +
		poseMs + triangulationMs + localBAMs;

	const std::string frameHeader =
		"timestamp,image_id,image_name,global_feature_ms,retrieval_ms,"
		"sift_extract_ms,matching_ms,pose_estimation_ms,triangulation_ms,"
		"local_ba_ms,sfm_total_ms,depth_total_ms,depth_alignment_ms,"
		"tsdf_integrate_ms,tsdf_extract_ms,frame_elapsed_ms";
	std::ostringstream frameRow;
	frameRow << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
		<< imageID << ","
		<< GpuDiagnostics::EscapeCsvValue(imageName) << ","
		<< globalFeatureMs << ","
		<< retrievalMs << ","
		<< siftMs << ","
		<< matchingMs << ","
		<< poseMs << ","
		<< triangulationMs << ","
		<< localBAMs << ","
		<< sfmTotalMs << ",,,,,";
	GpuDiagnostics::AppendTimingRow(
		"pipeline_frame_timing.csv", frameHeader, frameRow.str());

	const std::string sfmHeader =
		"timestamp,image_id,image_name,registered,num_registered_images,"
		"num_points3d,num_observations,new_points,filtered_points,"
		"global_feature_ms,retrieval_ms,sift_extract_ms,matching_ms,"
		"pose_estimation_ms,triangulation_ms,local_ba_ms,sfm_total_ms";
	std::ostringstream sfmRow;
	sfmRow << GpuDiagnostics::EscapeCsvValue(GpuDiagnostics::Timestamp()) << ","
		<< imageID << ","
		<< GpuDiagnostics::EscapeCsvValue(imageName) << ","
		<< (registered ? "true" : "false") << ","
		<< model.NumRegImages() << ","
		<< model.NumPoints3D() << ","
		<< model.ComputeNumObservations() << ","
		<< newPoints << ","
		<< filteredPoints << ","
		<< globalFeatureMs << ","
		<< retrievalMs << ","
		<< siftMs << ","
		<< matchingMs << ","
		<< poseMs << ","
		<< triangulationMs << ","
		<< localBAMs << ","
		<< sfmTotalMs;
	GpuDiagnostics::AppendTimingRow(
		"sfm_runtime_stats.csv", sfmHeader, sfmRow.str());
}

} // namespace

void ReconstructTask::run()
{
	taskStart = std::chrono::steady_clock::now();
	try {
		if (type == ReconstructType::PrepareRetry) {
			m_data->LoadCache(120, true);
			auto* mapper = m_data->getIncrementalMapper();
			auto model = m_data->getModelManager()->Get(0);
			QWriteLocker locker(&(m_data->m_models_lock[0]));
			reconstructionActive = true;
			mapper->BeginReconstruction(model);
			IncrementalMapper::Options options;
			retryBatch->candidates = mapper->FindNextImages(options);
			for (const auto id : m_data->EstimatedImageSnapshot()) {
				if (model->ExistsImage(id) && model->IsImageRegistered(id)) continue;
				++retryBatch->poolSize;
				if (!m_data->getDatabaseCache()->ExistsImage(id)) {
					++retryBatch->cacheMissing;
					retryBatch->waitingReasons[static_cast<image_t>(id)] = "no_qualifying_cached_matches";
				} else if (model->Image(id).NumVisiblePoints3D() < options.abs_pose_min_num_inliers) {
					++retryBatch->insufficientSupport;
					retryBatch->waitingReasons[static_cast<image_t>(id)] = "insufficient_visible_points3d";
				} else {
					retryBatch->waitingReasons[static_cast<image_t>(id)] = "pnp_not_registered";
				}
			}
			mapper->EndReconstruction(false);
			reconstructionActive = false;
		} else {
			if (!m_data->CheckCache(imageID) && type != ReconstructType::RetryBatch) m_data->LoadCache(120);
			if (type == ReconstructType::AddModel) {
				modelID = m_data->ModelManagerGetNum() == 0 ? m_data->ModelManagerAddModel() : 0;
				const bool success = Reconstruct(true);
				emit reconstructionInitial(modelID, success);
			} else if (type == ReconstructType::AddImage || type == ReconstructType::RetryBatch) {
				Reconstruct(false);
			}
		}
	} catch (const std::exception& e) {
		if (retryBatch) retryBatch->failed = true;
		LOG(ERROR) << "[Reconstruct] Image " << imageID << " task failed: " << e.what();
	} catch (...) {
		if (retryBatch) retryBatch->failed = true;
		LOG(ERROR) << "[Reconstruct] Image " << imageID << " task failed with unknown exception";
	}
	if (reconstructionActive) {
		QWriteLocker locker(&(m_data->m_models_lock[0]));
		m_data->getIncrementalMapper()->EndReconstruction(false);
		reconstructionActive = false;
	}
	emit reconstructTaskFinished(imageID);
}

bool ReconstructTask::Reconstruct(bool init)
{
	if (Config::getInstance().pipeline.OutputTimingFiles && !Database::exportPath.empty()) {
		GpuDiagnostics::SetOutputRoot(
			StringReplace(
				EnsureTrailingSlash(GetParentDir(Database::exportPath)),
				"\\",
				"/") + "DenseReconstruction");
	}
	Timer reconstructionTimer;
	reconstructionTimer.Start();

	IncrementalMapper* mapper = m_data->getIncrementalMapper();

	std::shared_ptr<Reconstruction> model = m_data->getModelManager()->Get(modelID);

	{
		QWriteLocker locker(&(m_data->m_models_lock[0]));
		reconstructionActive = true;
		mapper->BeginReconstruction(model);
	}

	IncrementalMapper::Options init_mapper_options;

	if (init) {
		// InitializeEmptyModel
		for (size_t numInitTrials = 0; numInitTrials < 2; numInitTrials++)
		{
			if (model->NumRegImages() > 0)
			{
				break;
			}
			// 只对空模型尝试初始化
			{
				QWriteLocker locker(&(m_data->m_models_lock[0]));
				if (InitializeEmptyModel(init_mapper_options, *model, *mapper))
				{
					//lastRegModelID = modelID;
					//Callback(CHANGE_CURRENT_MODEL_CALLBACK);
					break;
				}

				init_mapper_options.init_min_num_inliers /= 2;
				if (InitializeEmptyModel(init_mapper_options, *model, *mapper))
				{
					//lastRegModelID = modelID;
					//Callback(CHANGE_CURRENT_MODEL_CALLBACK);
					break;
				}
			}
			init_mapper_options.init_min_tri_angle /= 2;
		}
		if (model->NumRegImages() == 0)
		{
			mapper->EndReconstruction(true);
			reconstructionActive = false;
			return false;
		}
		size_t registerTime = 0;
		image_t lastFailedImageID = kInvalidImageId;
		size_t sameFailCount = 0;
		while (true)
		{
			std::vector<image_t> nextImageIDs = mapper->FindNextImages(init_mapper_options);
			if (nextImageIDs.empty())
			{
				break;
			}
			const image_t nextImageID = nextImageIDs.front();

			reconstructionTimer.Restart();
			const std::string nextImageName = m_data->DatabaseReadImage(nextImageID).Name();
			bool success = false;
			size_t poseMs = 0;
			size_t triangulationMs = 0;
			size_t localBAMs = 0;
			size_t newPoints = 0;
			{
				QWriteLocker locker(&(m_data->m_models_lock[0]));
				if (!mapper->RegisterNextImage(init_mapper_options, nextImageID))
				{
					model->reconstructTime += (reconstructionTimer.ElapsedMicroSeconds() / 1000);
					LOG(INFO) << "[model " << (modelID + 1) << "] image " << nextImageName << " registered failed!";
				}
				else
				{
					success = true;

					poseMs = reconstructionTimer.ElapsedMicroSeconds() / 1000;
					model->reconstructTime += poseMs;
					registerTime += poseMs;
					model->registerTimes.push_back(registerTime);
					LOG(INFO) << "[model " << (modelID + 1) << "] image " << nextImageName << " registered successfully!";

					reconstructionTimer.Restart();
					newPoints = TriangulateImage(init_mapper_options, nextImageID, *mapper);
					triangulationMs = reconstructionTimer.ElapsedMicroSeconds() / 1000;
					model->reconstructTime += triangulationMs;
					model->triangulateTimes.push_back(triangulationMs);

					reconstructionTimer.Restart();
					LocalBundleAdjustment(init_mapper_options, nextImageID, *mapper);
					localBAMs = reconstructionTimer.ElapsedMicroSeconds() / 1000;
					ExtractColors(init_mapper_options, nextImageID, *model);

					model->reconstructTime += (reconstructionTimer.ElapsedMicroSeconds() / 1000);
					AppendSfmTiming(
						m_data,
						nextImageID,
						nextImageName,
						*model,
						true,
						newPoints,
						0,
						static_cast<double>(poseMs),
						static_cast<double>(triangulationMs),
						static_cast<double>(localBAMs));
				}
			}

			if (!success)
			{
				if (lastFailedImageID == nextImageID)
				{
					++sameFailCount;
					if (sameFailCount >= 3)
					{
						LOG(WARNING) << "image " << nextImageName << " failed repeatedly, break to avoid dead loop.";
						break;
					}
				}
				else
				{
					lastFailedImageID = nextImageID;
					sameFailCount = 1;
				}
				continue;
			}
			lastFailedImageID = kInvalidImageId;
			sameFailCount = 0;

			m_data->AddReconstructedImage(nextImageID);
			emit registedSuccess(nextImageID);
		}
		model->reconstructTime += (reconstructionTimer.ElapsedMicroSeconds() / 1000);
	}
	else {
		// IncrementalMapping
		reconstructionTimer.Restart();
		size_t registerTime = 0;
		std::vector<image_t> nextImageIDs;
		if (type == ReconstructType::RetryBatch) {
			nextImageIDs = retryBatch->candidates;
		} else {
			if (model->ExistsImage(imageID) && !model->IsImageRegistered(imageID) &&
				model->Image(imageID).NumVisiblePoints3D() >= init_mapper_options.abs_pose_min_num_inliers) {
				nextImageIDs.push_back(imageID);
			}
		}
		registerTime += reconstructionTimer.ElapsedMicroSeconds() / 1000;
		model->reconstructTime += (reconstructionTimer.ElapsedMicroSeconds() / 1000);

		//if (nextImageIDs.empty()) {}

		std::vector<size_t> orderedNextImageIDs;
		orderedNextImageIDs.reserve(nextImageIDs.size());
		for (size_t id : nextImageIDs)
		{
			if (type == ReconstructType::RetryBatch || id == imageID)
			{
				orderedNextImageIDs.push_back(id);
			}
		}
		int itNum = 0;
		const auto batchStart = taskStart;
		for (size_t nextImageID : orderedNextImageIDs)
		{
			if (type == ReconstructType::RetryBatch) {
				if (retryBatch->consumed >= 5 || (retryBatch->consumed > 0 &&
					std::chrono::steady_clock::now() - batchStart >= std::chrono::seconds(2))) break;
				++retryBatch->consumed;
			}
			if (!model->ExistsImage(nextImageID) || model->IsImageRegistered(nextImageID)) continue;
			reconstructionTimer.Restart();
			const std::string nextImageName = m_data->DatabaseReadImage(nextImageID).Name();
			{
				QWriteLocker locker(&(m_data->m_models_lock[0]));
				if (retryBatch) ++retryBatch->attempted;
				if (!mapper->RegisterNextImage(init_mapper_options, nextImageID))
				{
					model->reconstructTime += (reconstructionTimer.ElapsedMicroSeconds() / 1000);
					LOG(INFO) << "[model " << (modelID + 1) << "] image " << nextImageName << " registered failed!";
					continue;
				}

				const size_t poseMs = reconstructionTimer.ElapsedMicroSeconds() / 1000;
				model->reconstructTime += poseMs;
				registerTime += poseMs;
				model->registerTimes.push_back(registerTime);
				LOG(INFO) << "[model " << (modelID + 1) << "] image " << nextImageName << " registered successfully!";

				reconstructionTimer.Restart();
				const size_t newPoints = TriangulateImage(init_mapper_options, nextImageID, *mapper);
				const size_t triangulationMs = reconstructionTimer.ElapsedMicroSeconds() / 1000;
				model->reconstructTime += triangulationMs;
				model->triangulateTimes.push_back(triangulationMs);

				reconstructionTimer.Restart();
				LocalBundleAdjustment(init_mapper_options, nextImageID, *mapper);
				const size_t localBAMs = reconstructionTimer.ElapsedMicroSeconds() / 1000;
				ExtractColors(init_mapper_options, nextImageID, *model);

				model->reconstructTime += (reconstructionTimer.ElapsedMicroSeconds() / 1000);
				AppendSfmTiming(
					m_data,
					nextImageID,
					nextImageName,
					*model,
					true,
					newPoints,
					0,
					static_cast<double>(poseMs),
					static_cast<double>(triangulationMs),
					static_cast<double>(localBAMs));
			}

			// 输出3DGS结果
			if (Export3DGS)
			{
				if (modelID != 0)
				{
					continue;
				}
				const size_t numRegImages = model->NumRegImages();
				const auto& new_points = GetImagePoint3D(model, nextImageID);		//获取当前影像所有可见3D点
				if (numRegImages % 1 == 0 && numRegImages != 0 && numRegImages > 1)
				{
					//std::thread exportProjectThread([&]() {
					const std::string basePath = StringReplace(EnsureTrailingSlash(GetParentDir(Database::exportPath)), "\\", "/");

					// 输出结果
					const std::string modelExportDir = basePath + "GaussianSplatting/" + std::to_string(numRegImages);
					CreateDirIfNotExists(modelExportDir + "/bin", true);
					CreateDirIfNotExists(modelExportDir + "/text", true);
					// 注册影像数量小于30的时候，输出所有点云
					if (IsInitialized3DGS == false)
					{
						model->WriteBinary(modelExportDir + "/bin");
						model->WriteText(modelExportDir + "/text");
						if (numRegImages >= 30)
						{
							IsInitialized3DGS = true;
						}
					}
					// 只输出更新的点云大于30的时候，输出新增的点云
					else
					{
						model->WriteBinaryFor3DGS(modelExportDir + "/bin", new_points, local_ba_images);
						model->WriteTextFor3GDS(modelExportDir + "/text", new_points, local_ba_images);
					}
					model->OutputDebugResult(modelExportDir);

					// 清空当前影像保存的新3D点NewPoints3D
					// mapper.ClearNewPoints3D();
					local_ba_images.clear();

					const size_t imageOutNum = numRegImages;
					const std::vector<image_t> RegImagesIds = model->RegImageIds();

					std::vector<std::pair<image_t, image_t>> imagePairs(0);
					std::vector<int> numInliers(0);
					m_data->getDatabase()->ReadTwoViewGeometryNumInliers(&imagePairs, &numInliers);
					for (int i = 0; i < imagePairs.size(); i++)
					{
						const image_pair_t imagePairID = Database::ImagePairToPairId(imagePairs[i].first, imagePairs[i].second);
						const size_t numInlierMatches = numInliers[i];
						matchMatrix[imagePairID] = numInlierMatches;
					}

					//输出影像名字
					std::ofstream MatchMatrixImageName(modelExportDir + "/match_matrix_image_name.txt");
					for (size_t i = 1; i <= imageOutNum; i++)
					{
						if (i < imageOutNum)
						{
							const image_t imageId = RegImagesIds[i - 1];
							MatchMatrixImageName << RemoveExtension(m_data->getDatabase()->ReadImage(imageId).Name()) << ",";
						}
						if (i == imageOutNum)
						{
							const image_t imageId = RegImagesIds[i - 1];
							MatchMatrixImageName << RemoveExtension(m_data->getDatabase()->ReadImage(imageId).Name()) << endl;
						}

					}
					MatchMatrixImageName.close();

					//输出匹配关系矩阵
					std::ofstream MatchMatrixValue(modelExportDir + "/match_matrix_value.txt");
					for (size_t i = 1; i <= imageOutNum; i++)
					{
						for (size_t j = 1; j <= imageOutNum; j++)
						{
							const image_t imageID1 = RegImagesIds[i - 1];
							const image_t imageID2 = RegImagesIds[j - 1];
							const size_t value = GetMatrixValue(imageID1, imageID2);
							if (j < imageOutNum)
							{
								if (j != i)
								{
									MatchMatrixValue << value << ",";
								}
								if (j == i)
								{
									MatchMatrixValue << m_data->getDatabase()->NumKeypointsForImage(imageID1) << ",";
								}
							}
							if (j == imageOutNum)
							{
								if (j != i)
								{
									MatchMatrixValue << value;
								}
								if (j == i)
								{
									MatchMatrixValue << m_data->getDatabase()->NumKeypointsForImage(imageID1);
								}
							}
						}
						MatchMatrixValue << endl;
					}

					//std::cout << (boost::format("Export %1% models completed!") % numRegImages).str() << endl;
					LOG(INFO) << (boost::format("Export %1% models completed!") % numRegImages).str();
				}
			}

			{
				if (model->NumRegImages() >= start &&
					(model->NumRegImages() - start) % step == 0 &&
					!m_data->m_feedback_lock) {
					QReadLocker locker(&(m_data->m_models_lock[0]));
					m_data->setRawData(std::make_shared<RawData>(CopyReconstruction(*model)));
				}
			}
			m_data->AddReconstructedImage(nextImageID);
			if (retryBatch) ++retryBatch->succeeded;
			emit registedSuccess(nextImageID);
		}
	}
	{
		QWriteLocker locker(&(m_data->m_models_lock[0]));
		mapper->EndReconstruction(false);
		reconstructionActive = false;
	}
	return true;
}

bool ReconstructTask::InitializeEmptyModel(const IncrementalMapper::Options& options, Reconstruction& model, IncrementalMapper& mapper)
{
	LOG(INFO) << "Finding good initial image pair...";

	image_t initImageID1 = std::numeric_limits<image_t>::max();
	image_t initImageID2 = std::numeric_limits<image_t>::max();
	const bool isFindInitSuccess = mapper.FindInitialImagePair(options, &initImageID1, &initImageID2);
	if (!isFindInitSuccess)
	{
		LOG(INFO) << "No good initial image pair found!";
		return false;
	}
	//if (!database->ExistsImage(initImageID1) || !database->ExistsImage(initImageID2))
	//{
	//	cout << (boost::format("Initial image pair %1% and %2% do not exist!") % initImageID1 % initImageID2).str() << endl;
	//	return false;
	//}

	LOG(INFO) << "Initializing with image pair " << initImageID1 << " and " << initImageID2 << "...";
	const bool isRegInitSuccess = mapper.RegisterInitialImagePair(options, initImageID1, initImageID2);
	if (!isRegInitSuccess)
	{
		LOG(INFO) << "Initialization failed!";
		return false;
	}
	LOG(INFO) << "Initialization successful!";

	GlobalBundleAdjustment(options, mapper);
	FilterPoints(options, mapper);
	FilterImages(options, mapper);

	if (model.NumRegImages() == 0 || model.NumPoints3D() == 0)
	{
		model.DeRegisterImage(initImageID1);
		model.DeRegisterImage(initImageID2);
		LOG(INFO) << "After bundle adjustment, the registered images or 3D points is empty!";
		return false;
	}
	ExtractColors(options, initImageID1, model);
	//Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

	m_data->AddReconstructedImage(initImageID1);
	m_data->AddReconstructedImage(initImageID2);
	emit registedInitialImagePair(initImageID1, initImageID2);
	return true;
}

void ReconstructTask::LocalBundleAdjustment(const IncrementalMapper::Options& options, size_t imageID, IncrementalMapper& mapper)
{
	BundleAdjustmentOptions custom_ba_options;
	custom_ba_options.ba_type = "Local BA";
	IncrementalMapper::Options localBundleAdjustOptions = options;
	IncrementalTriangulator::Options triOptions;

	const size_t numLocalBAIters = static_cast<size_t>(cfg.sfm.numLocalBAIters);

	std::vector<double> localBAErrors;
	std::vector<size_t> localBATimes;

	for (size_t i = 0; i < numLocalBAIters; i++)
	{
		Timer localBATimer;
		localBATimer.Start();

		double finalCost = 0;
		const IncrementalMapper::LocalBundleAdjustmentReport report = mapper.AdjustLocalBundle(localBundleAdjustOptions, custom_ba_options, triOptions, imageID, mapper.GetModifiedPoints3D(), finalCost);
		double changed = 0;
		if (report.num_adjusted_observations > 0)
		{
			changed = (report.num_merged_observations + report.num_completed_observations + report.num_filtered_observations) * 1.0 / report.num_adjusted_observations;
		}

		if (Export3DGS)
		{
			local_ba_images = report.local_ba_image_ids;
		}

#ifdef _DEBUG
		cout << (boost::format("[Reconstructor] Merged observations: %1%") % report.num_merged_observations).str() << endl;
		cout << (boost::format("[Reconstructor] Completed observations: %1%") % report.num_completed_observations).str() << endl;
		cout << (boost::format("[Reconstructor] Filtered observations: %1%") % report.num_filtered_observations).str() << endl;
		cout << (boost::format("[Reconstructor] Changed observations: %1%") % changed).str() << endl;
#endif

		const size_t localBATime = localBATimer.ElapsedMicroSeconds() / 1000;
		if (changed < 0.001f)
		{
			localBATimes.push_back(localBATime);
			localBAErrors.push_back(finalCost);
			break;
		}
		custom_ba_options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::TRIVIAL;

		localBATimes.push_back(localBATime);
		localBAErrors.push_back(finalCost);
		}
	mapper.ClearModifiedPoints3D();

	mapper.GetReconstruction()->localBAErrors.push_back(localBAErrors);
	mapper.GetReconstruction()->localBATimes.push_back(localBATimes);
	}

void ReconstructTask::GlobalBundleAdjustment(const IncrementalMapper::Options& options, IncrementalMapper& mapper)
{
	Timer globalBATimer;
	globalBATimer.Start();

	BundleAdjustmentOptions custom_ba_options;
	custom_ba_options.ba_type = "Global BA";
	IncrementalMapper::Options globalBundleAdjustOptions = options;
	if (mapper.GetReconstruction()->NumRegImages() < 10)
	{
		custom_ba_options.solver_options.function_tolerance /= 10;
		custom_ba_options.solver_options.gradient_tolerance /= 10;
		custom_ba_options.solver_options.parameter_tolerance /= 10;
		custom_ba_options.solver_options.max_num_iterations *= 2;
		custom_ba_options.solver_options.max_linear_solver_iterations = 200;
	}
	custom_ba_options.solver_options.function_tolerance = 0;
	custom_ba_options.solver_options.gradient_tolerance = 1.0;
	custom_ba_options.solver_options.parameter_tolerance = 0.0;
	custom_ba_options.solver_options.max_num_iterations = 50;
	custom_ba_options.solver_options.max_linear_solver_iterations = 100;
	custom_ba_options.solver_options.minimizer_progress_to_stdout = true;

	custom_ba_options.solver_options.minimizer_progress_to_stdout = true;
	custom_ba_options.solver_options.logging_type = ceres::LoggingType::SILENT;

	double finalCost = 0;
	mapper.AdjustGlobalBundle(globalBundleAdjustOptions, custom_ba_options, finalCost);
	LOG(INFO) << "[Global BA] final cost: " << finalCost << "px";

	mapper.GetReconstruction()->finalGlobalBATime = globalBATimer.ElapsedMicroSeconds() / 1000;
	mapper.GetReconstruction()->finalGloablBAError = finalCost;
}

size_t ReconstructTask::TriangulateImage(const IncrementalMapper::Options& options, size_t imageID, IncrementalMapper& mapper)
{
	IncrementalTriangulator::Options triOptions;
	const size_t numTris = mapper.TriangulateImage(triOptions, imageID);
	return numTris;
}

size_t ReconstructTask::FilterPoints(const IncrementalMapper::Options& options, IncrementalMapper& mapper)
{
	const size_t numFilterPoints = mapper.FilterPoints(options);
	return numFilterPoints;
}

size_t ReconstructTask::FilterImages(const IncrementalMapper::Options& options, IncrementalMapper& mapper)
{
	const size_t numFilterImages = mapper.FilterImages(options);
	return numFilterImages;
}

void ReconstructTask::ExtractColors(const IncrementalMapper::Options& options, size_t imageID, Reconstruction& model)
{
	const std::string imageDir = EnsureTrailingSlash(StringReplace(Database::imageDir, "\\", "/"));
	const std::string imagePath = imageDir + m_data->DatabaseReadImage(imageID).Name();
	if (model.ExtractColorsForImage(imageID, imageDir))
	{
		return;
	}
	LOG(INFO) << "[Reconstructor] Could not read image " << imagePath;
}

void ReconstructTask::CopyEssentialReconstruction()
{
	/*auto& modelptr = m_data->readModelPtr;
	if (modelptr) modelptr.reset();
	{
		QReadLocker locker(&(m_data->getModelLock(0)));
		std::shared_ptr<Reconstruction> model = m_data->getModelManager()->Get(0);
		auto start = std::chrono::high_resolution_clock::now();
		modelptr = model->CloneEssentialOnly();
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		LOG(INFO) << "DEBUG CloneEssentialOnly() took: " << duration.count() << " microseconds" << std::endl;
	}*/
}

//3DGS输出配套函数
size_t ReconstructTask::GetMatrixValue(image_t i, image_t j)
{
	if (i == j)
	{
		return 0;
	}
	const image_pair_t imagePairID = Database::ImagePairToPairId(i, j);
	const auto it = matchMatrix.find(imagePairID);
	if (it == matchMatrix.end())
	{
		return 0;
	}
	return it->second;
}

//去掉影像名的后缀
std::string ReconstructTask::RemoveExtension(std::string& filename)
{
	// 找到最后一个点的位置
	size_t dot_pos = filename.find_last_of(".");
	if (dot_pos != std::string::npos) {
		return filename.substr(0, dot_pos);  // 去掉后缀
	}
	return filename;  // 如果没有后缀，返回原始文件名
}

std::vector<point3D_t> ReconstructTask::GetImagePoint3D(const std::shared_ptr<Reconstruction> model, const image_t image_id)
{
	std::vector<point3D_t> point3D_ids;

	if (!model) {
		return point3D_ids;
	}

	if (!model->IsImageRegistered(image_id)) {
		return point3D_ids;
	}

	const Image& image = model->Image(image_id);

	point3D_ids.reserve(image.NumPoints2D());

	for (const Point2D& point2D : image.Points2D())
	{
		if (point2D.HasPoint3D())
		{
			point3D_ids.push_back(point2D.point3D_id);
		}
	}
	return point3D_ids;
}

