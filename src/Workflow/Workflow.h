// WorkFlow.h
#ifndef WORKFLOW_H
#define WORKFLOW_H

#include "../Base/config.h"
#include "../Scene/Database.h"
#include "../Scene/ReconstructionManager.h"
#include "../Scene/RenderData.h"
#include "DenseScaleInitTask.h"
#include "DenseFusionTask.h"
#include "DepthAlignmentTask.h"
#include "DepthTask.h"
#include "EstimateTask.h"
#include "ExtractTask.h"
#include "FeedbackTask.h"
#include "MatchTask.h"
#include "PythonTask.h"
#include "ReadTask.h"
#include "ReconstructTask.h"
#include "RetrieveTask.h"
#include "WorkData.h"
#include <QFuture>
#include <QMutex>
#include <QObject>
#include <QReadWriteLock>
#include <QString>
#include <QtConcurrent>
#include <QThread>
#include <QThreadPool>
#include <atomic>
#include <map>
#include <set>
#include <deque>
#include <QTimer>

class WorkFlow : public QObject
{
	Q_OBJECT
	friend struct SfMRegistrationTest;

public:
	explicit WorkFlow(QObject* parent = nullptr);
	~WorkFlow();

	void setData(Database* p_database, ReconstructionManager* p_modelManager);
	void LoadPython();
	void WarmupDepthWorker();
	void Shutdown();
	bool IsIdleForExperiment() const;
	void NotifyInputFinished(size_t expectedImages);
	void SuspendRegistrationRetry();
	bool IsRegistrationFinalized() const { return m_registrationFinalized; }
	bool IsInitializationReady() const;
	bool HasInitializationFailed() const;
	QString InitializationFailureMessage() const;
	int64_t LastActivityMs() const { return m_lastActivityMs.load(); }
	std::shared_ptr<const RenderData> GetRenderingData() const { return m_workData->GetRenderData(); }
	std::shared_ptr<const RenderData> GetDenseRenderingData() const { return m_workData->GetDenseRenderData(); }
	QReadWriteLock& GetModelLock(int modelID) { return m_workData->m_models_lock[modelID]; }
	void AcquireUsedFrame() { m_workData->usedFrame.acquire(); }
	void ReleaseFreeFrame() { m_workData->freeFrame.release(); }
	bool SpeedCtrl() const {
		const size_t limit = Config::getInstance().sfm.maxPendingImages;
		return limit > 0 &&
			m_workData->imagesInRcon.load() > limit &&
			m_workData->estimatedImagesNum.load() > 20;
	}

	static bool Export3DGS();

signals:
	void readImageFinished(QString p_imagePath);
	void reconstructUpdated(quint32 p_lastRegistedImageID); // refresh the model in model viewer
	void feedbackFinished();
	void updateSimulationSpeed(quint32 read, quint32 reconstructed);// 更新模拟器传入影像的速度
	void depthWorkerInitialized(QString message);
	void depthWorkerInitializeFailed(QString errorMessage);
	void denseScaleInitializedForViewer();
	void denseReconstructionUpdated();

public slots:
	void onProcessNewImage(const QString& value); // 传入影像，启动工作流
	void onReadTaskFinished(quint32 imageid, QString p_imagePath);
	void onLocalFeatureExtractFinished(QString p_imagePath);
	//void onLocalFeatureExtractFinished(quint32 imageID);
	void onGlobalFeatureExtractFinished(QString p_imagePath);
	void onRetrieveTaskFinished(quint32 imageID1, quint32 imageID2);
	//void onRetrieveTaskFinished(quint32 imageID);
	void onMatchTaskFinished(quint32 imageID1, quint32 imageID2);
	void onEstimateFinished(quint32 imageID1, quint32 imageID2);
	void onRegistedInitialImagePair(quint32 imageID1, quint32 imageID2);
	void onRegistedSuccess(quint32 imageID);
	void onFeedbackFinished();
	void onDepthPredictionFinished(quint32 imageID, QString depthPath, QString colorPath);
	void onDepthPredictionFailed(quint32 imageID, QString errorMessage);
	void onDepthAlignmentFinished(quint32 imageID, QString alignedDepthPath, QString colorPath);
	void onDepthAlignmentFailed(quint32 imageID, QString errorMessage);
	void onDenseScaleInitialized(bool success, QString message);
	void onDepthWorkerInitialized(QString message);
	void onDepthWorkerInitializeFailed(QString errorMessage);
	void onDenseFusionFinished(
		quint32 imageID,
		int integratedFrames,
		int vertices,
		int triangles,
		bool denseRenderUpdated);
	void onDenseFusionFailed(quint32 imageID, QString errorMessage);

private:
	void StartTrackedTask(QThreadPool* pool, BaseTask* task, bool frontEnd = false, int priority = 0);
	void SettleSfmImage(const QString& path);
	void CheckRegistrationRetry();
	void StartRegistrationRetry(bool finalRound);
	void ContinueRegistrationRetry();
	void QueueReconstruction(image_t id);
	void OnFrontEndFailed(const QString& path, const QString& reason);
	void OnMatchingPairFailed(image_t id1, image_t id2, const QString& reason);
	QTimer* m_registrationTimer = nullptr;
	size_t m_pendingTasks = 0;
	size_t m_pendingFrontEndTasks = 0;
	size_t m_pendingReconstructionTasks = 0;
	std::set<QString> m_firstPassPaths;
	std::set<image_t> m_firstPassScheduled;
	size_t m_receivedImages = 0;
	size_t m_expectedImages = 0;
	bool m_inputFinished = false;
	bool m_registrationFinalized = false;
	bool m_registrationFinalizationFailed = false;
	bool m_registrationRetrySuspended = false;
	bool m_retryActive = false;
	bool m_retryFinal = false;
	bool m_initializationQueued = false;
	size_t m_lastRetryRegistered = 0;
	size_t m_lastRetryRevision = 0;
	size_t m_retryStartRegistered = 0;
	size_t m_retryStartRevision = 0;
	int64_t m_lastRegistrationMs = 0;
	std::deque<image_t> m_retryCandidates;
	size_t m_retryAttempted = 0;
	size_t m_retrySucceeded = 0;
	bool m_retryFailed = false;
	std::map<image_t, std::string> m_retryWaitingReasons;
	WorkData* m_workData;      // 共享数据
	QThreadPool* m_read_threadPool; // 线程池
	QThreadPool* m_extract_threadPool;
	QThreadPool* m_gextract_threadPool;
	QThreadPool* m_retrieve_threadPool;
	QThreadPool* m_match_threadPool;
	QThreadPool* m_estimate_threadPool;
	QThreadPool* m_reconstruct_threadPool;
	QThreadPool* m_feedback_threadPool;
	QThreadPool* m_depth_threadPool;
	QThreadPool* m_depth_alignment_threadPool;
	QThreadPool* m_dense_fusion_threadPool;

	QMutex mu_reconstructTasks; // 分发任务加锁
	QMutex mu_depthTasks;
	QFuture<void> m_pythonLoadFuture;
	std::atomic_bool m_stopping{ false };
	std::atomic_bool m_shutdownComplete{ false };
	std::atomic_bool m_denseScaleInitStarted{ false };
	std::atomic_bool m_denseScaleInitialized{ false };
	std::atomic_bool m_pendingDepthAfterScaleInit{ false };
	std::atomic_bool m_globalFeatureReady{ false };
	std::atomic_bool m_depthWorkerReady{ false };
	std::atomic_bool m_initializationFailed{ false };
	QString m_initializationFailureMessage;
	mutable QMutex m_initializationMutex;
	std::set<image_t> scheduledDepthImages;

	bool ReconstructImage(quint32 imageID);
	void Feedback();
	bool EnsureDenseScaleInitialized();
	void PredictDepth(image_t imageID);
	void PredictDepthForRegisteredImages();
	void AlignDepth(image_t imageID, const QString& depthPath, const QString& colorPath);
	void FuseDepth(image_t imageID, const QString& depthPath, const QString& colorPath);
	//void TryStartMatching(quint32 imageID);

	static bool export3DGS_;

	void MarkActivity();

	std::atomic_int64_t m_lastActivityMs{ 0 };
};

#endif // WORKFLOW_H
