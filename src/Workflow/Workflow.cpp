#include "../Base/config.h"
#include "../Base/GpuDiagnostics.h"
#include "../Base/misc.h"
#include "glog/logging.h"
#include "WorkData.h"
#include "WorkFlow.h"

#include <algorithm>
#include <chrono>
#include <sstream>

using namespace std;

bool WorkFlow::export3DGS_ = false;

namespace {

const char* BoolToString(const bool value)
{
	return value ? "true" : "false";
}

std::string ToLower(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

std::string JoinStrings(
	const std::vector<std::string>& values,
	const char* delimiter = ",")
{
	std::ostringstream stream;
	for (size_t i = 0; i < values.size(); ++i) {
		if (i > 0) {
			stream << delimiter;
		}
		stream << values[i];
	}
	return stream.str();
}

std::string DenseDiagnosticsBasePath()
{
	if (!Database::exportPath.empty()) {
		return StringReplace(
			EnsureTrailingSlash(GetParentDir(Database::exportPath)),
			"\\",
			"/") + "DenseReconstruction";
	}
	return "";
}

void LogDenseConfiguration(const Config& cfg)
{
	LOG(INFO) << "[Config] SfM: useFocalLength="
		<< BoolToString(cfg.sfm.useFocalLength)
		<< ", cameraModel=" << cfg.sfm.cameraModel
		<< ", numLocalBAIters=" << cfg.sfm.numLocalBAIters
		<< ", maxNumFeatures=" << cfg.sfm.maxNumFeatures
		<< ", MaxPendingImages=" << cfg.sfm.maxPendingImages
		<< ", RegistrationRetryInterval=" << cfg.sfm.RegistrationRetryInterval
		<< ", RegistrationRetryIdleSeconds=" << cfg.sfm.RegistrationRetryIdleSeconds;

	LOG(INFO) << "[Config] Pipeline: AutoRunImages="
		<< BoolToString(cfg.pipeline.AutoRunImages)
		<< ", AutoExportModel=" << BoolToString(cfg.pipeline.AutoExportModel)
		<< ", InitialImagesNum=" << cfg.pipeline.InitialImagesNum
		<< ", Export3DGS=" << BoolToString(cfg.pipeline.Export3DGS)
		<< ", OutputTimingFiles=" << BoolToString(cfg.pipeline.OutputTimingFiles);
	const auto& experiment = cfg.experiment;
	if (!experiment.Enable) {
		LOG(INFO) << "[Config] Experiment: Enable=false";
	}
	else {
		LOG(INFO) << "[Config] Experiment: Enable=true"
			<< ", ProjectPath=" << experiment.ProjectPath
			<< ", ImagePath=" << experiment.ImagePath
			<< ", ShowUI=" << BoolToString(experiment.ShowUI)
			<< ", AutoSaveIdleSeconds=" << experiment.AutoSaveIdleSeconds;
	}

	const auto& denseDepth = cfg.denseDepth;
	if (!denseDepth.Enable) {
		LOG(INFO) << "[Config] DenseDepth: Enable=false";
	}
	else {
		const std::string method = denseDepth.Method.empty() ? "Murre" : denseDepth.Method;
		const std::string methodLower = ToLower(method);
		LOG(INFO) << "[Config] DenseDepth: Enable=true"
			<< ", Method=" << method
			<< ", Device=" << denseDepth.Device
			<< ", StartRegisteredImages=" << denseDepth.StartRegisteredImages
			<< ", TimeoutMs=" << denseDepth.TimeoutMs
			<< ", ExportSfMBin=" << BoolToString(denseDepth.ExportSfMBin)
			<< ", OutputResolution=" << denseDepth.OutputResolution;
		if (methodLower == "priordepthanything" || methodLower == "priorda" ||
			methodLower == "prior-depth-anything") {
			LOG(INFO) << "[Config] DenseDepth.PriorDepthAnything: ProcessingResolution="
				<< denseDepth.PriorDAProcessingResolution
				<< ", MaxDepth=" << denseDepth.PriorDAMaxDepth
				<< ", Version=" << denseDepth.PriorDAVersion
				<< ", FrozenModelSize=" << denseDepth.PriorDAFrozenModelSize
				<< ", ConditionedModelSize=" << denseDepth.PriorDAConditionedModelSize
				<< ", Pattern=" << denseDepth.PriorDAPattern;
		}
		else if (methodLower == "depthanythingv2" || methodLower == "depthanything-v2" ||
			methodLower == "depth-anything-v2" || methodLower == "dav2") {
			LOG(INFO) << "[Config] DenseDepth.DepthAnythingV2: ProcessingResolution="
				<< denseDepth.DepthAnythingV2ProcessingResolution
				<< ", MaxDepth=" << denseDepth.DepthAnythingV2MaxDepth
				<< ", Encoder=" << denseDepth.DepthAnythingV2Encoder
				<< ", Note=relative depth; enable DepthAlignment for TSDF";
		}
		else if (methodLower == "depthanything3" || methodLower == "depthanything-v3" ||
			methodLower == "depth-anything-3" || methodLower == "da3") {
			LOG(INFO) << "[Config] DenseDepth.DepthAnything3: Model="
				<< denseDepth.DepthAnything3Model
				<< ", ProcessingResolution=" << denseDepth.DepthAnything3ProcessingResolution
				<< ", ProcessResMethod=" << denseDepth.DepthAnything3ProcessResMethod
				<< ", MaxDepth=" << denseDepth.DepthAnything3MaxDepth
				<< ", ConfThreshold=" << denseDepth.DepthAnything3ConfThreshold;
		}
		else if (methodLower == "pixelperfectdepth" || methodLower == "pixel-perfect-depth" ||
			methodLower == "ppd") {
			LOG(INFO) << "[Config] DenseDepth.PixelPerfectDepth: SamplingSteps="
				<< denseDepth.PixelPerfectDepthSamplingSteps
				<< ", ProcessingResolution=" << denseDepth.PixelPerfectDepthProcessingResolution
				<< ", SemanticsModel=" << denseDepth.PixelPerfectDepthSemanticsModel
				<< ", MaxDepth=" << denseDepth.PixelPerfectDepthMaxDepth
				<< ", FP16=" << BoolToString(denseDepth.PixelPerfectDepthFP16);
		}
		else if (methodLower == "moge" || methodLower == "moge2" || methodLower == "moge-2") {
			LOG(INFO) << "[Config] DenseDepth.MoGe: Model=" << denseDepth.MoGeModel
				<< ", ResolutionLevel=" << denseDepth.MoGeResolutionLevel
				<< ", NumTokens=" << denseDepth.MoGeNumTokens
				<< ", MaxDepth=" << denseDepth.MoGeMaxDepth
				<< ", FP16=" << BoolToString(denseDepth.MoGeUseFP16)
				<< ", UseSfMFov=" << BoolToString(denseDepth.MoGeUseSfMFov)
				<< ", ApplyMask=" << BoolToString(denseDepth.MoGeApplyMask)
				<< ", MaskThreshold=" << denseDepth.MoGeMaskThreshold
				<< ", SaveNormal=" << BoolToString(denseDepth.MoGeSaveNormal);
		}
		else {
			LOG(INFO) << "[Config] DenseDepth.Murre: ProcessingResolution="
				<< denseDepth.MurreProcessingResolution
				<< ", MaxDepth=" << denseDepth.MurreMaxDepth
				<< ", DenoiseSteps=" << denseDepth.MurreDenoiseSteps
				<< ", EnsembleSize=" << denseDepth.MurreEnsembleSize;
		}
	}

	const auto& depthAlignment = cfg.depthAlignment;
	if (!depthAlignment.Enable) {
		LOG(INFO) << "[Config] DepthAlignment: Enable=false";
	}
	else {
		LOG(INFO) << "[Config] DepthAlignment: Enable=true"
			<< ", SaveAlignDepth=" << BoolToString(depthAlignment.SaveAlignDepth)
			<< ", Mode=" << depthAlignment.Mode
			<< ", MinPoints=" << depthAlignment.MinPoints
			<< ", MaxDepth=" << depthAlignment.MaxDepth
			<< ", Grid=" << depthAlignment.Grid
			<< ", Overlap=" << depthAlignment.Overlap
			<< ", LocalFit=" << BoolToString(depthAlignment.LocalFit)
			<< ", LocalModelMode=" << depthAlignment.LocalModelMode
			<< ", MinBlockPoints=" << depthAlignment.MinBlockPoints
			<< ", MaxBlockSamples=" << depthAlignment.MaxBlockSamples;
	}

	const auto& weightedTSDF = cfg.weightedTSDF;
	if (!weightedTSDF.Enable) {
		LOG(INFO) << "[Config] WeightedTSDF: Enable=false";
	}
	else {
		LOG(INFO) << "[Config] WeightedTSDF: Enable=true"
			<< ", Algorithm=GlobalP75AlignGrid"
			<< ", GlobalP75Metric=" << weightedTSDF.GlobalP75Metric
			<< ", GlobalResidualReferenceQuantile=0.75"
			<< ", AlignmentWeightGridMatched=true"
			<< ", MinUpgradeSamples=" << weightedTSDF.MinUpgradeSamples
			<< ", UnsupportedRegionScale=" << weightedTSDF.UnsupportedRegionScale
			<< ", HardRejectRatio=" << weightedTSDF.HardRejectRatio
			<< ", SoftSuppressionEnable=" << BoolToString(weightedTSDF.SoftSuppressionEnable)
			<< ", SoftSuppressionExponent=" << weightedTSDF.SoftSuppressionExponent
			<< ", MaxVoxelWeight=" << weightedTSDF.MaxVoxelWeight
			<< ", ExtractionWeightThreshold=" << weightedTSDF.ExtractionWeightThreshold
			<< ", UpgradeCapDivisor=" << weightedTSDF.UpgradeCapDivisor
			<< ", SaveQualityMaps=" << BoolToString(weightedTSDF.SaveQualityMaps);
	}

	const auto& denseFusion = cfg.denseFusion;
	if (!denseFusion.Enable) {
		LOG(INFO) << "[Config] DenseFusion: Enable=false";
	}
	else {
		LOG(INFO) << "[Config] DenseFusion: Enable=true"
			<< ", Device=" << denseFusion.Device
			<< ", OutputMode=" << denseFusion.OutputMode
			<< ", ColorMode=" << denseFusion.ColorMode
			<< ", VoxelLength=" << denseFusion.VoxelLength
			<< ", TruncVoxelMultiplier=" << denseFusion.TruncVoxelMultiplier
			<< ", DepthMax=" << denseFusion.DepthMax
			<< ", ExtractEveryNFrames=" << denseFusion.ExtractEveryNFrames
			<< ", MaxDisplayPoints=" << denseFusion.MaxDisplayPoints
			<< ", MinValidDepthPixels=" << denseFusion.MinValidDepthPixels;
	}

	LOG(INFO) << "[Config] BundleAdjustment: LocalBACudaMinObservations="
		<< cfg.bundleAdjustment.LocalBACudaMinObservations;

	const auto& gpuDiagnostics = cfg.gpuDiagnostics;
	if (!gpuDiagnostics.Enable) {
		LOG(INFO) << "[Config] GpuDiagnostics: Enable=false";
	}
	else {
		LOG(INFO) << "[Config] GpuDiagnostics: Enable=true"
			<< ", PollIntervalMs=" << gpuDiagnostics.PollIntervalMs
			<< ", UseNvidiaSmi=" << BoolToString(gpuDiagnostics.UseNvidiaSmi)
			<< ", LogEventSnapshots=" << BoolToString(gpuDiagnostics.LogEventSnapshots)
			<< ", LogBundleAdjustmentStats=" << BoolToString(gpuDiagnostics.LogBundleAdjustmentStats);
	}
}

} // namespace

WorkFlow::WorkFlow(QObject* parent)
	: QObject(parent)
	, m_workData(new WorkData(this))
	, m_read_threadPool(new QThreadPool(this))
	, m_extract_threadPool(new QThreadPool(this))
	, m_gextract_threadPool(new QThreadPool(this))
	, m_retrieve_threadPool(new QThreadPool(this))
	, m_match_threadPool(new QThreadPool(this))
	, m_estimate_threadPool(new QThreadPool(this))
	, m_reconstruct_threadPool(new QThreadPool(this))
	, m_feedback_threadPool(new QThreadPool(this))
	, m_depth_threadPool(new QThreadPool(this))
	, m_depth_alignment_threadPool(new QThreadPool(this))
	, m_dense_fusion_threadPool(new QThreadPool(this))
{
	m_lastActivityMs.store(std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count());
	const auto& cfg = Config::getInstance();
	#ifdef NOT_USE_GLOBAL_FEATURES
	m_globalFeatureReady.store(true);
	#endif
	if (!cfg.denseDepth.Enable) {
		m_depthWorkerReady.store(true);
	}
	export3DGS_ = cfg.pipeline.Export3DGS;
	LogDenseConfiguration(cfg);
	GpuDiagnostics::SetOutputRoot(DenseDiagnosticsBasePath());
	GpuDiagnostics::RecordCurrentProcess("UI");
	GpuDiagnostics::StartProcessSampler();

	LoadPython();
	m_read_threadPool->setMaxThreadCount(1);
	m_extract_threadPool->setMaxThreadCount(2);
	m_gextract_threadPool->setMaxThreadCount(1);
	m_retrieve_threadPool->setMaxThreadCount(2);
	m_match_threadPool->setMaxThreadCount(3);
	m_estimate_threadPool->setMaxThreadCount(6);
	m_reconstruct_threadPool->setMaxThreadCount(1);
	m_feedback_threadPool->setMaxThreadCount(1);
	m_depth_threadPool->setMaxThreadCount(1);
	m_depth_threadPool->setExpiryTimeout(-1);
	m_depth_alignment_threadPool->setMaxThreadCount(1);
	m_depth_alignment_threadPool->setExpiryTimeout(-1);
	m_dense_fusion_threadPool->setMaxThreadCount(1);
	m_dense_fusion_threadPool->setExpiryTimeout(-1);
	m_lastRegistrationMs = m_lastActivityMs.load();
	m_registrationTimer = new QTimer(this);
	connect(m_registrationTimer, &QTimer::timeout, this, &WorkFlow::CheckRegistrationRetry);
	m_registrationTimer->start(1000);
}

bool WorkFlow::IsInitializationReady() const
{
	return m_globalFeatureReady.load() &&
		m_depthWorkerReady.load();
}

bool WorkFlow::HasInitializationFailed() const
{
	return m_initializationFailed.load();
}

QString WorkFlow::InitializationFailureMessage() const
{
	QMutexLocker locker(&m_initializationMutex);
	return m_initializationFailureMessage;
}

void WorkFlow::MarkActivity()
{
	m_lastActivityMs.store(std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count());
}

void WorkFlow::StartTrackedTask(QThreadPool* pool, BaseTask* task, bool frontEnd, int priority)
{
	++m_pendingTasks;
	if (frontEnd) ++m_pendingFrontEndTasks;
	connect(task, &BaseTask::frontEndFailed, this, &WorkFlow::OnFrontEndFailed);
	connect(task, &BaseTask::matchingPairFailed, this, &WorkFlow::OnMatchingPairFailed);
	// Auto-deletion happens after run(), including early returns and caught failures.
	connect(task, &QObject::destroyed, this, [this, frontEnd]() {
		--m_pendingTasks;
		if (frontEnd) --m_pendingFrontEndTasks;
		MarkActivity();
	}, Qt::QueuedConnection);
	pool->start(task, priority);
}

void WorkFlow::SettleSfmImage(const QString& path)
{
	if (m_firstPassPaths.erase(path)) m_workData->imagesInRcon.fetch_sub(1);
}

void WorkFlow::OnFrontEndFailed(const QString& path, const QString& reason)
{
	if (m_stopping.load()) return;
	LOG(WARNING) << "[SfM] Front-end terminal failure: " << path.toStdString()
		<< ", reason=" << reason.toStdString();
	SettleSfmImage(path);
	try {
		const auto id = m_workData->getImageID(path.toStdString());
		m_workData->AddEstimatedImage(id);
	} catch (const std::out_of_range&) {
		// Reading failed before an image ID was assigned.
	}
}

void WorkFlow::OnMatchingPairFailed(image_t id1, image_t id2, const QString& reason)
{
	LOG(WARNING) << "[SfM] Pair " << id1 << "-" << id2 << " settled as failed: " << reason.toStdString();
	if (m_workData->CompleteMatchingPair(id1, id2)) {
		m_workData->AddEstimatedImage(id1);
		m_workData->AddEstimatedImage(id2);
		onEstimateFinished(id1, id2);
	}
}

void WorkFlow::NotifyInputFinished(size_t expectedImages)
{
	if (m_stopping.load()) return;
	expectedImages = std::max(expectedImages, m_receivedImages);
	if (m_inputFinished && m_expectedImages == expectedImages) {
		CheckRegistrationRetry();
		return;
	}
	m_inputFinished = true;
	m_expectedImages = expectedImages;
	m_registrationRetrySuspended = false;
	m_registrationFinalized = false;
	CheckRegistrationRetry();
}

void WorkFlow::SuspendRegistrationRetry()
{
	m_registrationRetrySuspended = true;
	m_inputFinished = false;
	m_registrationFinalized = false;
}

void WorkFlow::CheckRegistrationRetry()
{
	if (m_stopping.load() || m_registrationRetrySuspended || m_retryActive || m_initializationQueued) return;
	const bool drained = m_inputFinished && m_receivedImages >= m_expectedImages &&
		m_pendingFrontEndTasks == 0 && m_pendingReconstructionTasks == 0;
	const bool initialized = m_workData->reconstructImagesNum.load() > 0;
	if (!initialized) {
		if (drained && !m_initializationQueued) {
			if (!m_registrationFinalized) LOG(WARNING) << "[SfM Retry] Input drained without an initialized model.";
			m_registrationFinalized = true;
			for (const auto& path : std::set<QString>(m_firstPassPaths)) SettleSfmImage(path);
		}
		return;
	}
	const size_t registered = m_workData->reconstructImagesNum.load();
	const size_t revision = m_workData->sfmRevision.load();
	const bool changed = registered != m_lastRetryRegistered || revision != m_lastRetryRevision;
	if (m_registrationFinalizationFailed && !changed) return;
	if (drained && m_registrationFinalized) return;
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
	const auto& sfm = Config::getInstance().sfm;
	if (drained || (changed && (registered - std::min(registered, m_lastRetryRegistered) >= sfm.RegistrationRetryInterval ||
		now - m_lastRegistrationMs >= static_cast<int64_t>(sfm.RegistrationRetryIdleSeconds) * 1000))) {
		StartRegistrationRetry(drained);
	}
}

void WorkFlow::StartRegistrationRetry(bool finalRound)
{
	m_retryActive = true;
	m_registrationFinalizationFailed = false;
	m_retryFinal = finalRound;
	m_retryFailed = false;
	m_retryStartRegistered = m_workData->reconstructImagesNum.load();
	m_retryStartRevision = m_workData->sfmRevision.load();
	m_retryAttempted = m_retrySucceeded = 0;
	auto batch = std::make_shared<RegistrationRetryBatch>();
	auto task = new ReconstructTask(m_workData, 0, 0, ReconstructType::PrepareRetry, 0, 1);
	task->SetRetryBatch(batch);
	connect(task, &ReconstructTask::reconstructTaskFinished, this, [this, batch](quint32) {
		m_retryFailed = batch->failed;
		m_retryWaitingReasons = batch->waitingReasons;
		m_retryCandidates.assign(batch->candidates.begin(), batch->candidates.end());
		LOG(INFO) << "[SfM Retry] Prepared: final=" << m_retryFinal << ", pool=" << batch->poolSize
			<< ", candidates=" << batch->candidates.size() << ", insufficient_support=" << batch->insufficientSupport
			<< ", cache_missing=" << batch->cacheMissing;
		ContinueRegistrationRetry();
	});
	StartTrackedTask(m_reconstruct_threadPool, task);
}

void WorkFlow::ContinueRegistrationRetry()
{
	if (m_stopping.load()) return;
	if (m_registrationRetrySuspended) {
		m_retryCandidates.clear();
		m_retryActive = false;
		return;
	}
	if (m_retryCandidates.empty() || m_retryFailed) {
		m_lastRetryRegistered = m_retryStartRegistered;
		m_lastRetryRevision = m_retryStartRevision;
		m_retryActive = false;
		LOG(INFO) << "[SfM Retry] Round finished: attempted=" << m_retryAttempted
			<< ", succeeded=" << m_retrySucceeded << ", failed=" << m_retryFailed;
		if (m_retryFinal && (m_retryFailed || (m_workData->reconstructImagesNum.load() == m_retryStartRegistered &&
			m_workData->sfmRevision.load() == m_retryStartRevision))) {
			m_registrationFinalized = !m_retryFailed;
			m_registrationFinalizationFailed = m_retryFailed;
			if (m_retryFailed) LOG(ERROR) << "[SfM Retry] Finalization failed; automatic export is blocked.";
			if (m_registrationFinalized) {
				for (const auto& entry : m_retryWaitingReasons) {
					if (!m_workData->IsReconstructedImage(entry.first)) {
						LOG(INFO) << "[SfM Retry] Unregistered image " << entry.first << ": " << entry.second;
					}
				}
			}
		}
		return;
	}
	auto batch = std::make_shared<RegistrationRetryBatch>();
	for (size_t i = 0; i < 5 && i < m_retryCandidates.size(); ++i) batch->candidates.push_back(m_retryCandidates[i]);
	auto task = new ReconstructTask(m_workData, 0, 0, ReconstructType::RetryBatch,
		Config::getInstance().feedback.StartCreateMesh, Config::getInstance().feedback.Step);
	task->SetRetryBatch(batch);
	connect(task, &ReconstructTask::registedSuccess, this, &WorkFlow::onRegistedSuccess, Qt::BlockingQueuedConnection);
	connect(task, &ReconstructTask::reconstructTaskFinished, this, [this, batch](quint32) {
		for (size_t i = 0; i < batch->consumed; ++i) m_retryCandidates.pop_front();
		m_retryAttempted += batch->attempted;
		m_retrySucceeded += batch->succeeded;
		m_retryFailed = batch->failed;
		ContinueRegistrationRetry();
	});
	StartTrackedTask(m_reconstruct_threadPool, task);
}

bool WorkFlow::IsIdleForExperiment() const
{
	if (m_pendingTasks != 0 || m_retryActive || m_pendingReconstructionTasks != 0) return false;
	// imagesInRcon is a registration backlog counter used by SpeedCtrl().
	// Failed/unregistered images intentionally remain in that counter, so it
	// cannot be used to decide whether all scheduled work has finished.
	const QThreadPool* pools[] = {
		m_read_threadPool, m_extract_threadPool, m_gextract_threadPool,
		m_retrieve_threadPool, m_match_threadPool, m_estimate_threadPool,
		m_reconstruct_threadPool, m_feedback_threadPool, m_depth_threadPool,
		m_depth_alignment_threadPool, m_dense_fusion_threadPool,
	};
	for (const QThreadPool* pool : pools) {
		if (pool && pool->activeThreadCount() != 0) {
			return false;
		}
	}
	return true;
}

WorkFlow::~WorkFlow()
{
	Shutdown();
	delete m_read_threadPool;
	delete m_extract_threadPool;
	delete m_gextract_threadPool;
	delete m_retrieve_threadPool;
	delete m_match_threadPool;
	delete m_estimate_threadPool;
	delete m_reconstruct_threadPool;
	delete m_feedback_threadPool;
	delete m_depth_threadPool;
	delete m_depth_alignment_threadPool;
	delete m_dense_fusion_threadPool;
	delete m_workData;
}

void WorkFlow::Shutdown()
{
	GpuDiagnostics::StopProcessSampler();
	bool expected = false;
	if (!m_shutdownComplete.compare_exchange_strong(expected, true)) {
		return;
	}

	m_stopping.store(true);
	if (m_registrationTimer) m_registrationTimer->stop();
	disconnect(nullptr, nullptr, this, nullptr);

	auto stopPool = [](QThreadPool* pool) {
		if (!pool) {
			return;
		}
		pool->clear();
		pool->waitForDone();
	};

	stopPool(m_read_threadPool);
	stopPool(m_extract_threadPool);
	stopPool(m_gextract_threadPool);
	stopPool(m_retrieve_threadPool);
	stopPool(m_match_threadPool);
	stopPool(m_estimate_threadPool);
	stopPool(m_reconstruct_threadPool);
	stopPool(m_feedback_threadPool);
	stopPool(m_depth_threadPool);
	stopPool(m_depth_alignment_threadPool);
	stopPool(m_dense_fusion_threadPool);

	if (m_pythonLoadFuture.isRunning()) {
		m_pythonLoadFuture.waitForFinished();
	}
}

void WorkFlow::setData(Database* p_database, ReconstructionManager* p_modelManager)
{
	m_workData->Initial(p_database, p_modelManager);
}

void WorkFlow::LoadPython()
{
	m_pythonLoadFuture = QtConcurrent::run([this]() {
		try {
			QThread::sleep(1);
			if (m_stopping.load()) {
				return;
			}
			PyLoader* pyloader = new PyLoader();
			size_t retrievalTopN = 30;
			CGlobalFeatureExtractor* globalFeatureExtractor = nullptr;
			CGlobalFeatureRetriever* globalFeatureRetriever = nullptr;
#ifndef NOT_USE_GLOBAL_FEATURES
			globalFeatureExtractor = new CGlobalFeatureExtractor();
			globalFeatureRetriever = new CGlobalFeatureRetriever(m_workData->getDatabase(), retrievalTopN);
			if (!globalFeatureExtractor->IsInitialized()) {
				QMutexLocker locker(&m_initializationMutex);
				m_initializationFailureMessage = "Global feature extractor was not initialized.";
				m_initializationFailed.store(true);
				delete globalFeatureExtractor;
				delete globalFeatureRetriever;
				return;
			}
#endif
			if (m_stopping.load()) {
				delete globalFeatureExtractor;
				delete globalFeatureRetriever;
				return;
			}
			m_workData->setGlobalFeatureFunc(globalFeatureExtractor, globalFeatureRetriever);
			m_globalFeatureReady.store(true);
		}
		catch (const std::exception& e) {
			{
				QMutexLocker locker(&m_initializationMutex);
				m_initializationFailureMessage = QString("Global feature initialization failed: %1").arg(e.what());
			}
			m_initializationFailed.store(true);
			LOG(ERROR) << "[Python] LoadPython failed: " << e.what();
		}
		catch (...) {
			{
				QMutexLocker locker(&m_initializationMutex);
				m_initializationFailureMessage = "Global feature initialization failed: unknown exception.";
			}
			m_initializationFailed.store(true);
			LOG(ERROR) << "[Python] LoadPython failed with unknown exception.";
		}
		}
	);
}

void WorkFlow::WarmupDepthWorker()
{
	if (m_stopping.load()) {
		return;
	}
	if (!Config::getInstance().denseDepth.Enable) {
		return;
	}

	DepthTask* warmupTask = new DepthTask(m_workData);
	connect(warmupTask, &DepthTask::depthWorkerInitialized,
		this, &WorkFlow::onDepthWorkerInitialized);
	connect(warmupTask, &DepthTask::depthWorkerInitializeFailed,
		this, &WorkFlow::onDepthWorkerInitializeFailed);
	StartTrackedTask(m_depth_threadPool, warmupTask);
}


void WorkFlow::onProcessNewImage(const QString& value) {
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	GpuDiagnostics::SetOutputRoot(DenseDiagnosticsBasePath());
	++m_receivedImages;
	m_registrationRetrySuspended = false;
	if (m_inputFinished && m_receivedImages > m_expectedImages) m_inputFinished = false;
	m_registrationFinalized = false;
	if (!m_firstPassPaths.insert(value).second) return;
	m_workData->imagesInRcon.fetch_add(1);
	ReadTask* m_readTask = new ReadTask(m_workData, value);
	connect(m_readTask, &ReadTask::readTaskDone, this, &WorkFlow::onReadTaskFinished);
	StartTrackedTask(m_read_threadPool, m_readTask, true);
}

void WorkFlow::onReadTaskFinished(quint32 imageid, QString p_imagePath)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	emit readImageFinished(p_imagePath);
	ExtractTask* m_extractTask = new ExtractTask(m_workData, p_imagePath, imageid);
	connect(m_extractTask, &ExtractTask::localExtractFinished, this, &WorkFlow::onLocalFeatureExtractFinished);
	StartTrackedTask(m_extract_threadPool, m_extractTask, true);
}

//void WorkFlow::onReadTaskFinished(quint32 imageid, QString p_imagePath)
//{
//	emit readImageFinished(p_imagePath);
//
//	// Local feature extraction
//	ExtractTask* extractTask = new ExtractTask(m_workData, p_imagePath, imageid, this);
//	connect(extractTask, &ExtractTask::localExtractFinished, this, &WorkFlow::onLocalFeatureExtractFinished);
//	m_extract_threadPool->start(extractTask);
//
//	// Global feature extraction
//	PythonTask* gextractTask = new PythonTask(m_workData, p_imagePath, PythonTaskMode::GlobalFeatureExtract, this);
//	connect(gextractTask, &PythonTask::globalFeatureExtractFinished, this, &WorkFlow::onGlobalFeatureExtractFinished);
//	m_gextract_threadPool->start(gextractTask);
//}

void WorkFlow::onLocalFeatureExtractFinished(QString p_imagePath)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	PythonTask* m_gextractTask = new PythonTask(m_workData, p_imagePath, PythonTaskMode::GlobalFeatureExtract);
	connect(m_gextractTask, &PythonTask::globalFeatureExtractFinished, this, &WorkFlow::onGlobalFeatureExtractFinished);
	StartTrackedTask(m_gextract_threadPool, m_gextractTask, true);
}

//void WorkFlow::onLocalFeatureExtractFinished(QString p_imagePath)
//{
//	std::string imagePath = p_imagePath.toStdString();
//	quint32 imageID = m_workData->getImageID(imagePath);
//
//	m_workData->SetLocalReady(imageID);
//	//TryStartMatching(imageID);
//}

void WorkFlow::onGlobalFeatureExtractFinished(QString p_imagePath)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	RetrieveTask* m_retrieveTask = new RetrieveTask(m_workData, p_imagePath);
	connect(m_retrieveTask, &RetrieveTask::retrieveTaskFinished, this, &WorkFlow::onRetrieveTaskFinished);
	connect(m_retrieveTask, &RetrieveTask::retrievalCompleted, this, [this](quint32 id, bool success, bool hasPairs) {
		if (id != kInvalidImageId && !hasPairs) {
			m_workData->AddEstimatedImage(id);
			if (success) ReconstructImage(id);
		}
	});
	StartTrackedTask(m_retrieve_threadPool, m_retrieveTask, true);
}

void WorkFlow::onRetrieveTaskFinished(quint32 imageID1, quint32 imageID2) {
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	MatchTask* m_matchTask = new MatchTask(m_workData, imageID1, imageID2);
	connect(m_matchTask, &MatchTask::matchTaskFinished, this, &WorkFlow::onMatchTaskFinished);
	StartTrackedTask(m_match_threadPool, m_matchTask, true);
}

//void WorkFlow::onRetrieveTaskFinished(quint32 imageID)
//{
//	m_workData->SetRetrievalReady(imageID);
//	TryStartMatching(imageID);
//}
//
//void WorkFlow::TryStartMatching(quint32 imageID)
//{
//	const auto retrievedIDs = m_workData->GetRetrievedIDs(imageID);
//
//	for (const auto retrievedID : retrievedIDs)
//	{
//		MatchTask* matchTask = new MatchTask(m_workData, retrievedID, imageID, this);
//		connect(matchTask, &MatchTask::matchTaskFinished, this, &WorkFlow::onMatchTaskFinished);
//		m_match_threadPool->start(matchTask);
//	}
//}

void WorkFlow::onMatchTaskFinished(quint32 imageID1, quint32 imageID2)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	EstimateTask* m_estimateTask = new EstimateTask(m_workData, imageID1, imageID2);
	connect(m_estimateTask, &EstimateTask::estimateAllFinished, this, &WorkFlow::onEstimateFinished);
	StartTrackedTask(m_estimate_threadPool, m_estimateTask, true);
}

void WorkFlow::onEstimateFinished(quint32 imageID1, quint32 imageID2) {
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	QMutexLocker locker(&mu_reconstructTasks);
	ReconstructImage(imageID2);
}

void WorkFlow::onRegistedInitialImagePair(quint32 imageID1, quint32 imageID2) {
	if (m_stopping.load()) {
		return;
	}
	m_lastRegistrationMs = m_lastActivityMs.load();
	SettleSfmImage(QString::fromStdString(m_workData->getImagePath(imageID1)));
	SettleSfmImage(QString::fromStdString(m_workData->getImagePath(imageID2)));
	PredictDepthForRegisteredImages();
	emit reconstructUpdated(imageID2);
	emit updateSimulationSpeed(
		m_workData->readImagesNum.load(),
		m_workData->reconstructImagesNum.load()
	);
}

void WorkFlow::onRegistedSuccess(quint32 imageID) {
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	// feedback task
	PredictDepthForRegisteredImages();
	Feedback();
	emit reconstructUpdated(imageID);
	emit updateSimulationSpeed(
		m_workData->readImagesNum.load(),
		m_workData->reconstructImagesNum.load()
	);
	m_lastRegistrationMs = m_lastActivityMs.load();
	SettleSfmImage(QString::fromStdString(m_workData->getImagePath(imageID)));
}

void WorkFlow::onFeedbackFinished()
{
	if (m_stopping.load()) {
		return;
	}
	emit feedbackFinished();
}

void WorkFlow::onDepthPredictionFinished(quint32 imageID, QString depthPath, QString colorPath)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	LOG(INFO) << "[DenseDepth] Finished image " << imageID
		<< ", depth: " << depthPath.toStdString()
		<< ", color: " << colorPath.toStdString();
	if (Config::getInstance().depthAlignment.Enable) {
		AlignDepth(imageID, depthPath, colorPath);
	}
	else {
		FuseDepth(imageID, depthPath, colorPath);
	}
}

void WorkFlow::onDepthPredictionFailed(quint32 imageID, QString errorMessage)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	LOG(ERROR) << "[DenseDepth] Failed image " << imageID
		<< ": " << errorMessage.toStdString();
}

void WorkFlow::onDepthAlignmentFinished(quint32 imageID, QString alignedDepthPath, QString colorPath)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	LOG(INFO) << "[DepthAlignment] Finished image " << imageID
		<< ", aligned depth: " << alignedDepthPath.toStdString();
	FuseDepth(imageID, alignedDepthPath, colorPath);

}

void WorkFlow::onDepthAlignmentFailed(quint32 imageID, QString errorMessage)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	LOG(WARNING) << "[DepthAlignment] Skipping TSDF for image " << imageID
		<< ": " << errorMessage.toStdString();
	
}

void WorkFlow::onDenseScaleInitialized(bool success, QString message)
{
	if (m_stopping.load()) {
		return;
	}
	if (!success) {
		LOG(ERROR) << "[DenseScale] " << message.toStdString();
		return;
	}

	m_denseScaleInitialized.store(true);
	m_workData->sfmRevision.fetch_add(1);
	m_registrationFinalized = false;
	m_pendingDepthAfterScaleInit.store(false);
	LOG(INFO) << "[DenseScale] " << message.toStdString();
	emit denseScaleInitializedForViewer();
	PredictDepthForRegisteredImages();
}

void WorkFlow::onDepthWorkerInitialized(QString message)
{
	if (m_stopping.load()) {
		return;
	}
	m_depthWorkerReady.store(true);
	LOG(INFO) << "[DenseDepth] " << message.toStdString();
	emit depthWorkerInitialized(message);
}

void WorkFlow::onDepthWorkerInitializeFailed(QString errorMessage)
{
	if (m_stopping.load()) {
		return;
	}
	{
		QMutexLocker locker(&m_initializationMutex);
		m_initializationFailureMessage = QString("Dense depth initialization failed: %1").arg(errorMessage);
	}
	m_initializationFailed.store(true);
	LOG(ERROR) << "[DenseDepth] " << errorMessage.toStdString();
	emit depthWorkerInitializeFailed(QString("Dense depth model failed: %1").arg(errorMessage));
}



void WorkFlow::onDenseFusionFinished(
	quint32 imageID,
	int integratedFrames,
	int vertices,
	int triangles,
	bool denseRenderUpdated)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	LOG(INFO) << "[DenseFusion] Finished image " << imageID
		<< ", frames: " << integratedFrames
		<< ", vertices: " << vertices
		<< ", triangles: " << triangles
		<< ", dense render updated: " << BoolToString(denseRenderUpdated);
	if (denseRenderUpdated) {
		emit denseReconstructionUpdated();
	}
}

void WorkFlow::onDenseFusionFailed(quint32 imageID, QString errorMessage)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	LOG(ERROR) << "[DenseFusion] Failed image " << imageID
		<< ": " << errorMessage.toStdString();
}







bool WorkFlow::ReconstructImage(quint32 imageID)
{
	MarkActivity();
	if (m_stopping.load()) {
		return false;
	}
	QueueReconstruction(imageID);
	return false;
}

void WorkFlow::QueueReconstruction(image_t id)
{
	const auto& cfg = Config::getInstance();
	if (m_stopping.load() || m_workData->estimatedImagesNum.load() < cfg.pipeline.InitialImagesNum) return;
	if (m_workData->IsReconstructedImage(id)) {
		SettleSfmImage(QString::fromStdString(m_workData->getImagePath(id)));
		return;
	}
	const bool init = m_workData->reconstructImagesNum.load() == 0;
	if (init && m_initializationQueued) return;
	if (!m_firstPassScheduled.insert(id).second) return;
	m_initializationQueued = m_initializationQueued || init;
	++m_pendingReconstructionTasks;
	auto task = new ReconstructTask(m_workData, id, 0,
		init ? ReconstructType::AddModel : ReconstructType::AddImage, cfg.feedback.StartCreateMesh, cfg.feedback.Step);
	connect(task, &ReconstructTask::registedInitialImagePair, this, &WorkFlow::onRegistedInitialImagePair);
	connect(task, &ReconstructTask::registedSuccess, this, &WorkFlow::onRegistedSuccess, Qt::BlockingQueuedConnection);
	connect(task, &ReconstructTask::reconstructTaskFinished, this, [this, init](quint32 imageID) {
		--m_pendingReconstructionTasks;
		SettleSfmImage(QString::fromStdString(m_workData->getImagePath(imageID)));
		if (init) {
			m_initializationQueued = false;
			if (m_workData->reconstructImagesNum.load() > 0) {
				for (const auto id : m_workData->EstimatedImageSnapshot()) QueueReconstruction(static_cast<image_t>(id));
			}
		}
	});
	StartTrackedTask(m_reconstruct_threadPool, task);
}

void WorkFlow::Feedback()
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	auto& config = Config::getInstance();
	int startFeedback = config.feedback.StartCreateMesh;
	int step = config.feedback.Step;
	const size_t reconstructedImageNum = m_workData->reconstructImagesNum.load();
	if (m_workData->m_feedback_lock) return;
	if (reconstructedImageNum > startFeedback &&
		(reconstructedImageNum - startFeedback) % step == 0) {
		FeedbackTask* m_feedbackTask = new FeedbackTask(m_workData, 0);
		connect(m_feedbackTask, &FeedbackTask::taskFinished, this, &WorkFlow::onFeedbackFinished);
		StartTrackedTask(m_feedback_threadPool, m_feedbackTask);
	}
}

bool WorkFlow::EnsureDenseScaleInitialized()
{
	MarkActivity();
	if (m_stopping.load()) {
		return false;
	}
	const Config& config = Config::getInstance();
	if (!config.denseDepth.Enable) {
		return false;
	}
	const int startRegisteredImages =
		std::max(1, config.denseDepth.StartRegisteredImages);
	if (m_workData->reconstructImagesNum.load() < static_cast<size_t>(startRegisteredImages)) {
		return false;
	}
	if (m_denseScaleInitialized.load()) {
		return true;
	}

	m_pendingDepthAfterScaleInit.store(true);
	bool expected = false;
	if (m_denseScaleInitStarted.compare_exchange_strong(expected, true)) {
		LOG(INFO) << "[DenseScale] Queued global BA + Normalize before DenseDepth, priority=1. "
			<< "Registered images=" << m_workData->reconstructImagesNum.load()
			<< ", threshold=" << startRegisteredImages
			<< ", Normalize extent=10, p0=0.1, p1=0.9, use_images=true.";
		DenseScaleInitTask* denseScaleTask = new DenseScaleInitTask(m_workData);
		connect(denseScaleTask, &DenseScaleInitTask::denseScaleInitialized,
			this, &WorkFlow::onDenseScaleInitialized);
		StartTrackedTask(m_reconstruct_threadPool, denseScaleTask, false, 1);
	}
	return false;
}

void WorkFlow::PredictDepth(image_t imageID)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	if (!Config::getInstance().denseDepth.Enable) {
		return;
	}
	const int startRegisteredImages =
		std::max(1, Config::getInstance().denseDepth.StartRegisteredImages);
	if (m_workData->reconstructImagesNum.load() < static_cast<size_t>(startRegisteredImages)) {
		return;
	}
	if (!EnsureDenseScaleInitialized()) {
		return;
	}

	{
		QMutexLocker locker(&mu_depthTasks);
		if (scheduledDepthImages.count(imageID) > 0) {
			return;
		}
		scheduledDepthImages.insert(imageID);
	}

	DepthTask* depthTask = new DepthTask(m_workData, imageID);
	connect(depthTask, &DepthTask::depthPredictionFinished,
		this, &WorkFlow::onDepthPredictionFinished);
	connect(depthTask, &DepthTask::depthPredictionFailed,
		this, &WorkFlow::onDepthPredictionFailed);
	StartTrackedTask(m_depth_threadPool, depthTask);
}

void WorkFlow::PredictDepthForRegisteredImages()
{
	if (m_stopping.load()) {
		return;
	}
	if (!Config::getInstance().denseDepth.Enable) {
		return;
	}
	const int startRegisteredImages =
		std::max(1, Config::getInstance().denseDepth.StartRegisteredImages);
	if (m_workData->reconstructImagesNum.load() < static_cast<size_t>(startRegisteredImages)) {
		return;
	}
	if (!EnsureDenseScaleInitialized()) {
		return;
	}

	std::vector<image_t> regImageIds;
	{
		QReadLocker locker(&(m_workData->m_models_lock[0]));
		if (m_workData->getModelManager()->Size() == 0) {
			return;
		}
		const std::shared_ptr<Reconstruction>& model = m_workData->getModelManager()->Get(0);
		if (!model) {
			return;
		}
		regImageIds = model->RegImageIds();
	}

	for (const image_t imageID : regImageIds) {
		PredictDepth(imageID);
	}
}

void WorkFlow::AlignDepth(image_t imageID, const QString& depthPath, const QString& colorPath)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	DepthAlignmentTask* alignmentTask =
		new DepthAlignmentTask(m_workData, imageID, depthPath, colorPath);
	connect(alignmentTask, &DepthAlignmentTask::depthAlignmentFinished,
		this, &WorkFlow::onDepthAlignmentFinished);
	connect(alignmentTask, &DepthAlignmentTask::depthAlignmentFailed,
		this, &WorkFlow::onDepthAlignmentFailed);
	StartTrackedTask(m_depth_alignment_threadPool, alignmentTask);
}

void WorkFlow::FuseDepth(image_t imageID, const QString& depthPath, const QString& colorPath)
{
	MarkActivity();
	if (m_stopping.load()) {
		return;
	}
	if (!Config::getInstance().denseFusion.Enable) {
		return;
	}

	DenseFusionTask* fusionTask = new DenseFusionTask(m_workData, imageID, depthPath, colorPath);
	connect(fusionTask, &DenseFusionTask::denseFusionFinished,
		this, &WorkFlow::onDenseFusionFinished);
	connect(fusionTask, &DenseFusionTask::denseFusionFailed,
		this, &WorkFlow::onDenseFusionFailed);
	StartTrackedTask(m_dense_fusion_threadPool, fusionTask);
}





bool WorkFlow::Export3DGS()
{
	return export3DGS_;
}

