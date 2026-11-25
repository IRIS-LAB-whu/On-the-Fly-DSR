#include "../Base/config.h"
#include "../Base/macros.h"
#include "../Base/misc.h"
#include "../Workflow/DenseFusionTask.h"
#include "MainWindow.h"
#include <filesystem>
#include <fstream>
#include <QSettings>
#include <QString>
#include <QTabWidget>
#include <QMessageBox>
#include <QCoreApplication>
#include <QDir>
#include <QDateTime>
#include <QFileInfoList>
#include <algorithm>
#include <condition_variable>
#include <chrono>

using namespace std;

// ================= 工具函数 =================
namespace {
	std::string ToLowerCase(const std::string& str) {
		std::string lowerCaseStr = str;
		std::transform(lowerCaseStr.begin(), lowerCaseStr.end(), lowerCaseStr.begin(),
			[](unsigned char c) { return std::tolower(c); });
		return lowerCaseStr;
	}

	std::vector<std::string> GetImageFileList(const std::string& path) {
		std::vector<std::string> fileList;
		std::filesystem::path dirPath(path);
		const std::vector<std::string> imageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".gif" };

		if (std::filesystem::exists(dirPath) && std::filesystem::is_directory(dirPath)) {
			for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
				if (entry.is_regular_file()) {
					std::filesystem::path filePath = entry.path();
					std::string extension = ToLowerCase(filePath.extension().string());
					if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end()) {
						fileList.push_back(filePath.string());
					}
				}
			}
		}
		std::sort(fileList.begin(), fileList.end());
		return fileList;
	}

	QString StdString2QString2(std::string string) {
		return QString::fromLocal8Bit(string.data());
	}

	QString LatestRuntimeLogPath() {
		const QDir logDir(QDir(QCoreApplication::applicationDirPath()).filePath("../log"));
		const QFileInfoList logs = logDir.entryInfoList(
			QStringList() << "*.log", QDir::Files, QDir::Time);
		return logs.isEmpty() ? QString() : logs.front().absoluteFilePath();
	}

	int Add(int currentTime) {
		int hours = currentTime / 10000;
		int minutes = (currentTime / 100) % 100;
		int seconds = currentTime % 100;

		seconds++;
		if (seconds == 60) {
			seconds = 0;
			minutes++;
			if (minutes == 60) {
				minutes = 0;
				hours++;
				if (hours == 24) {
					hours = 0;
				}
			}
		}
		int newTime = hours * 10000 + minutes * 100 + seconds;
		return newTime;
	}

	int ToPlyColor(const float value) {
		return static_cast<int>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
	}

	bool SaveDensePointCloudPly(const RenderData& data, const std::string& path) {
		if (data.dense_points.empty()) {
			return false;
		}

		std::ofstream file(path);
		if (!file) {
			throw std::runtime_error("Cannot write dense point cloud PLY: " + path);
		}

		file << "ply\n"
			<< "format ascii 1.0\n"
			<< "element vertex " << data.dense_points.size() << "\n"
			<< "property float x\n"
			<< "property float y\n"
			<< "property float z\n"
			<< "property uchar red\n"
			<< "property uchar green\n"
			<< "property uchar blue\n"
			<< "end_header\n";

		for (const PointPainterData& point : data.dense_points) {
			file << point[0] << ' ' << point[1] << ' ' << point[2] << ' '
				<< ToPlyColor(point[3]) << ' '
				<< ToPlyColor(point[4]) << ' '
				<< ToPlyColor(point[5]) << '\n';
		}
		return true;
	}

	bool SaveDenseMeshPly(const RenderData& data, const std::string& path) {
		if (data.dense_mesh_data.empty()) {
			return false;
		}

		std::ofstream file(path);
		if (!file) {
			throw std::runtime_error("Cannot write dense mesh PLY: " + path);
		}

		const size_t numTriangles = data.dense_mesh_data.size();
		const size_t numVertices = numTriangles * 3;
		file << "ply\n"
			<< "format ascii 1.0\n"
			<< "element vertex " << numVertices << "\n"
			<< "property float x\n"
			<< "property float y\n"
			<< "property float z\n"
			<< "property uchar red\n"
			<< "property uchar green\n"
			<< "property uchar blue\n"
			<< "element face " << numTriangles << "\n"
			<< "property list uchar int vertex_indices\n"
			<< "end_header\n";

		for (const TrianglePainterData& triangle : data.dense_mesh_data) {
			for (int vertex = 0; vertex < 3; ++vertex) {
				const int offset = vertex * 7;
				file << triangle[offset] << ' '
					<< triangle[offset + 1] << ' '
					<< triangle[offset + 2] << ' '
					<< ToPlyColor(triangle[offset + 3]) << ' '
					<< ToPlyColor(triangle[offset + 4]) << ' '
					<< ToPlyColor(triangle[offset + 5]) << '\n';
			}
		}

		for (size_t i = 0; i < numTriangles; ++i) {
			const size_t vertexID = i * 3;
			file << "3 " << vertexID << ' ' << vertexID + 1 << ' ' << vertexID + 2 << '\n';
		}
		return true;
	}

	void SaveDenseReconstructionPly(
		const std::shared_ptr<const RenderData>& data,
		const std::string& denseDir) {
		if (ExportDenseFusionFullPly(denseDir)) {
			return;
		}

		if (!data) {
			cout << "No dense reconstruction data to export." << endl;
			return;
		}

		CreateDirIfNotExists(denseDir, true);
		const std::string normalizedDir = StringReplace(EnsureTrailingSlash(denseDir), "\\", "/");
		const bool savedPointCloud =
			SaveDensePointCloudPly(*data, normalizedDir + "dense_cloud.ply");
		const bool savedMesh =
			SaveDenseMeshPly(*data, normalizedDir + "dense_mesh.ply");

		if (savedPointCloud) {
			cout << "Dense point cloud saved to " << normalizedDir + "dense_cloud.ply" << endl;
		}
		if (savedMesh) {
			cout << "Dense mesh saved to " << normalizedDir + "dense_mesh.ply" << endl;
		}
		if (!savedPointCloud && !savedMesh) {
			cout << "No dense point cloud or mesh to export." << endl;
		}
	}

}

// ================= 构造/析构与事件 =================
MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
	const auto& experiment = Config::getInstance().experiment;
	if (experiment.Enable) {
		Database::imageDir = experiment.ImagePath;
		Database::exportPath = experiment.ProjectPath;
	}
	receiver = new Receiver(this);
	database = new Database();
	reconstructionManager = new ReconstructionManager();

	SetupWidgets();
	SetupActions();
	CreateMenus();
	CreateToolbar();
	CreateStatusbar();

	workflow = new WorkFlow(this);
	workflow->setData(database, reconstructionManager);

	SetupConnections();
	workflow->WarmupDepthWorker();

	if (experiment.Enable) {
		experimentIdleTimer = new QTimer(this);
		experimentIdleTimer->setInterval(1000);
		connect(experimentIdleTimer, &QTimer::timeout,
			this, &MainWindow::CheckExperimentIdle);
		QTimer::singleShot(0, this, &MainWindow::StartExperimentMode);
	}
	if (!experiment.ShowUI) {
		hide();
	}
}

MainWindow::~MainWindow()
{
	if (receiver) {
		receiver->Stop();
	}
	stopExperimentRequested.store(true);
	experimentCondition.notify_all();
	StopExperimentThread();
	if (exportThread.joinable()) {
		exportThread.join();
	}
	if (workflow) {
		workflow->Shutdown();
		delete workflow;
		workflow = nullptr;
	}
	delete receiver;
	receiver = nullptr;
	delete database;
	database = nullptr;
	delete reconstructionManager;
	reconstructionManager = nullptr;
}

void MainWindow::closeEvent(QCloseEvent* event) {
	stopExperimentRequested.store(true);
	experimentCondition.notify_all();
	if (receiver) {
		receiver->Stop();
	}
	StopExperimentThread();
	if (experimentIdleTimer) {
		experimentIdleTimer->stop();
	}
	if (exportThread.joinable()) {
		exportThread.join();
	}
	if (workflow) {
		workflow->Shutdown();
	}
	event->accept();
}

// ================= 初始化相关 =================
void MainWindow::SetupWidgets() {
	resize(800, 600);
	setWindowTitle(tr("On-the-fly SfM"));
	setWindowIcon(QIcon(":/media/WindowIcon.png"));

	tabWidget = new QTabWidget(this);
	modelViewerWidget = new ModelViewerWidget(this);
	modelViewerWidget->SetDatabase(database);
	tabWidget->addTab(modelViewerWidget, "Model Viewer");
	setCentralWidget(tabWidget);
	connect(modelViewerWidget, &ModelViewerWidget::statusUpdate, this, &MainWindow::UpdateStatus);

	reconstructionManagerWidget = new ReconstructionManagerWidget(this, std::shared_ptr<const ReconstructionManager>(reconstructionManager));
	reconstructionManagerWidget->setFixedWidth(270);
	reconstructionManagerWidget->Update();
	connect(reconstructionManagerWidget, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &MainWindow::ChangeSelectModel);

	imageListWindowDock = new QDockWidget(tr("Image Preview"), this);
	imageListWindowDock->setMinimumHeight(100);
	imageListWindowDock->setMinimumWidth(250);
	imageListWidget = new ImageListWidget(imageListWindowDock, database);
	addDockWidget(Qt::RightDockWidgetArea, imageListWindowDock);

	logWindowDock = new QDockWidget(tr("Log"), this);
	logWidget = new LogWidget(this);
	logWindowDock->setWidget(logWidget);
	addDockWidget(Qt::BottomDockWidgetArea, logWindowDock);
	logWindowDock->show();
	logWindowDock->raise();

	projectWidget = new ProjectWidget(this);
	connect(projectWidget, &ProjectWidget::NewProject_SIGNAL, this, &MainWindow::NewProject_SLOT);

	isTrackModelCheckBox = new QCheckBox(tr("Track Model"));
	isTrackModelCheckBox->setChecked(true);
	connect(isTrackModelCheckBox, &QCheckBox::toggled,
		this, &MainWindow::OnTrackModelToggled);

	showMatchMatrixWidget = new ShowMatchMatrixWidget(this, database);
}

void MainWindow::SetupActions() {
	newProjectAction = new QAction(QIcon(":/media/project-new.png"), tr("New Project"), this);
	connect(newProjectAction, &QAction::triggered, this, &MainWindow::NewProject);

	exportProjectAction = new QAction(QIcon(":/media/project-save-as.png"), tr("Export Project"), this);
	connect(exportProjectAction, &QAction::triggered, this, &MainWindow::ExportProject);
	exportProjectAction->setEnabled(false);

	startReceiveAction = new QAction(QIcon(":/media/reconstruction-start.png"), tr("Start Receive"), this);
	connect(startReceiveAction, &QAction::triggered, this, &MainWindow::StartReceive);

	stopReceiverAction = new QAction(QIcon(":/media/reconstruction-pause.png"), tr("Stop Receive"), this);
	connect(stopReceiverAction, &QAction::triggered, this, &MainWindow::StopReceive);
	stopReceiverAction->setEnabled(false);

	renderNowAction = new QAction(this);
	connect(renderNowAction, &QAction::triggered, this, &MainWindow::RenderNow);

	showMatchMatrixAction = new QAction(QIcon(":/media/match-matrix.png"), tr("Show Match Matrix"), this);
	connect(showMatchMatrixAction, &QAction::triggered, this, &MainWindow::ShowMatchMatrix);

	// 添加Feedback相关的Actions
	showMeshAction = new QAction(tr("Show Mesh"), this);
	showMeshAction->setCheckable(true);
	showMeshAction->setChecked(false);
	connect(showMeshAction, &QAction::triggered, this, &MainWindow::OnShowMeshChanged);

	showQualityAction = new QAction(tr("Show Quality"), this);
	showQualityAction->setCheckable(true);
	showQualityAction->setChecked(false);
	connect(showQualityAction, &QAction::triggered, this, &MainWindow::OnShowQualityChanged);

	showGuidePathAction = new QAction(tr("Show Guide Path"), this);
	showGuidePathAction->setCheckable(true);
	showGuidePathAction->setChecked(false);
	connect(showGuidePathAction, &QAction::triggered, this, &MainWindow::OnShowGuidePathChanged);

	denseViewAction = new QAction(QIcon(":/media/dense-reconstruction.png"), tr("Dense View"), this);
	denseViewAction->setCheckable(true);
	denseViewAction->setChecked(false);
	connect(denseViewAction, &QAction::triggered, this, &MainWindow::OnDenseViewChanged);

	const std::string denseOutputMode = ToLowerCase(Config::getInstance().denseFusion.OutputMode);
	const bool denseOutputsPoints =
		denseOutputMode == "pointcloud" || denseOutputMode == "point" ||
		denseOutputMode == "points" || denseOutputMode == "both";
	const bool denseOutputsMesh =
		denseOutputMode == "mesh" || denseOutputMode == "both";

	showDensePointAction = new QAction(tr("Show Dense Point"), this);
	showDensePointAction->setCheckable(true);
	showDensePointAction->setChecked(denseOutputsPoints);
	connect(showDensePointAction, &QAction::triggered, this, &MainWindow::OnShowDensePointChanged);

	showDenseMeshAction = new QAction(tr("Show Mesh"), this);
	showDenseMeshAction->setCheckable(true);
	showDenseMeshAction->setChecked(denseOutputsMesh);
	connect(showDenseMeshAction, &QAction::triggered, this, &MainWindow::OnShowDenseMeshChanged);


}

void MainWindow::CreateMenus() {
	projectMenu = new QMenu(tr("Project"), this);
	projectMenu->addAction(newProjectAction);
	projectMenu->addAction(exportProjectAction);
	menuBar()->addAction(projectMenu->menuAction());

	transmitMenu = new QMenu(tr("Transmit"), this);
	transmitMenu->addAction(startReceiveAction);
	transmitMenu->addAction(stopReceiverAction);
	menuBar()->addAction(transmitMenu->menuAction());

	feedbackMenu = new QMenu(tr("Feedback"), this);
	feedbackMenu->addAction(showMeshAction);
	feedbackMenu->addAction(showQualityAction);
	feedbackMenu->addAction(showGuidePathAction);
	menuBar()->addAction(feedbackMenu->menuAction());

	denseMenu = new QMenu(tr("Dense"), this);
	denseMenu->addAction(showDensePointAction);
	denseMenu->addAction(showDenseMeshAction);
	denseMenu->addSeparator();
	menuBar()->addAction(denseMenu->menuAction());

}

void MainWindow::CreateToolbar() {
	projectToolBar = addToolBar(tr("Project"));
	projectToolBar->addAction(newProjectAction);
	projectToolBar->addAction(exportProjectAction);
	projectToolBar->setIconSize(QSize(16, 16));
	projectToolBar->setMovable(false);

	transmitToolBar = addToolBar(tr("Transmit"));
	transmitToolBar->addAction(startReceiveAction);
	transmitToolBar->addAction(stopReceiverAction);
	transmitToolBar->setIconSize(QSize(16, 16));
	transmitToolBar->setMovable(false);

	reconstructToolBar = addToolBar(tr("Reconstruct"));
	reconstructToolBar->addAction(showMatchMatrixAction);
	reconstructToolBar->addWidget(reconstructionManagerWidget);
	reconstructToolBar->setIconSize(QSize(16, 16));
	reconstructToolBar->addSeparator();
	reconstructToolBar->addWidget(isTrackModelCheckBox);
	reconstructToolBar->addAction(denseViewAction);
	reconstructToolBar->setMovable(false);

	feedbackToolBar = addToolBar(tr("Feedback"));
	feedbackToolBar->setIconSize(QSize(16, 16));
	feedbackToolBar->setMovable(false);
}

void MainWindow::CreateStatusbar() {
	QFont font;
	font.setPointSize(11);
	statusLabel = new QLabel("0 Images - 0 Points", this);
	statusLabel->setFont(font);
	statusLabel->setAlignment(Qt::AlignCenter);
	statusBar()->addWidget(statusLabel, 1);
}

void MainWindow::SetupConnections() const {
	connect(receiver, &Receiver::newDataReceived, workflow, &WorkFlow::onProcessNewImage);
	connect(receiver, &Receiver::newDataReceived, this, &MainWindow::onProcessNewImage);
	connect(workflow, &WorkFlow::readImageFinished, this, &MainWindow::addNewImageInWidget);
	connect(workflow, &WorkFlow::reconstructUpdated, this, &MainWindow::OnRenderNow);
	connect(workflow, &WorkFlow::denseScaleInitializedForViewer, this, &MainWindow::OnDenseScaleInitializedForViewer);
	connect(workflow, &WorkFlow::feedbackFinished, this, &MainWindow::OnFeedback);
	connect(workflow, &WorkFlow::updateSimulationSpeed, this, &MainWindow::UpdateSimulationSpeed);
	connect(workflow, &WorkFlow::denseReconstructionUpdated, this, &MainWindow::OnDenseReconstructionUpdated);
}

// ================= 业务逻辑 =================
void MainWindow::RunFunc() {
	const std::string baseImagePath = Config::getInstance().experiment.Enable
		? Config::getInstance().experiment.ImagePath
		: Database::imageDir;
	const vector<std::string> images = GetImageFileList(baseImagePath);

	QSettings settings("HKEY_CURRENT_USER\\Software\\RTPS\\CamFiTransmit", QSettings::NativeFormat);
	{
		std::unique_lock<std::mutex> lock(experimentMutex);
		experimentCondition.wait_for(lock, chrono::milliseconds(200), [this]() {
			return stopExperimentRequested.load();
		});
	}
	for (int i = 0; i < images.size(); i++) {
		if (stopExperimentRequested.load()) {
			break;
		}
		const QString imagePath = StdString2QString2(images[i]);
		settings.setValue("NewImage", imagePath);
		if (i == 0) {
			imageSubmissionStarted.store(true);
		}
		while (workflow && workflow->SpeedCtrl()) {
			std::unique_lock<std::mutex> lock(experimentMutex);
			if (experimentCondition.wait_for(lock, chrono::milliseconds(500), [this]() {
				return stopExperimentRequested.load();
			})) {
				break;
			}
		}
		if (stopExperimentRequested.load()) {
			break;
		}
		std::unique_lock<std::mutex> lock(experimentMutex);
		experimentCondition.wait_for(lock, chrono::milliseconds(1000), [this]() {
			return stopExperimentRequested.load();
		});
	}
	if (!stopExperimentRequested.load()) {
		allImagesSubmitted.store(true);
		QMetaObject::invokeMethod(this, [this, expected = images.size()]() {
			if (workflow && !stopExperimentRequested.load()) workflow->NotifyInputFinished(expected);
		}, Qt::QueuedConnection);
		cout << "[Pipeline] All images submitted." << endl;
	}
	else {
		allImagesSubmitted.store(false);
		cout << "[Pipeline] Image sender stopped by user." << endl;
	}
}

void MainWindow::StartExperimentMode() {
	const auto& experiment = Config::getInstance().experiment;
	if (!experiment.Enable) {
		return;
	}
	if (experimentStarted.load()) {
		return;
	}
	const std::filesystem::path projectPath(experiment.ProjectPath);
	const std::filesystem::path imagePath(experiment.ImagePath);
	if (experiment.ProjectPath.empty() || projectPath.extension() != ".db" ||
		std::filesystem::exists(projectPath) ||
		!std::filesystem::is_directory(imagePath) ||
		GetImageFileList(experiment.ImagePath).empty()) {
		LOG(ERROR) << "[Experiment] Invalid configuration or existing project database. "
			<< "ProjectPath=" << experiment.ProjectPath
			<< ", ImagePath=" << experiment.ImagePath;
		return;
	}
	if (workflow && workflow->HasInitializationFailed()) {
		experimentStartPending.store(false);
		LOG(ERROR) << "[Experiment] Required model initialization failed: "
			<< workflow->InitializationFailureMessage().toStdString();
		return;
	}
	if (!workflow || !workflow->IsInitializationReady()) {
		if (!stopExperimentRequested.load()) {
			experimentStartPending.store(true);
			QTimer::singleShot(250, this, &MainWindow::StartExperimentMode);
		}
		return;
	}
	experimentStartPending.store(false);
	experimentStarted.store(true);
	cout << "[Experiment] Required models initialized." << endl;
	cout << "[Experiment] Starting image submission." << endl;
	startReceiveAction->setEnabled(false);
	stopReceiverAction->setEnabled(true);
	stopExperimentRequested.store(false);
	allImagesSubmitted.store(false);
	imageSubmissionStarted.store(false);
	autoExportStarted.store(false);
	autoExportFinished.store(false);
	experimentLogPath = LatestRuntimeLogPath();
	receiver->Start();
	if (experimentIdleTimer) {
		experimentIdleTimer->start();
	}
	experimentThread = std::thread(&MainWindow::RunFunc, this);
}

void MainWindow::StopExperimentThread() {
	if (experimentThread.joinable()) {
		experimentCondition.notify_all();
		experimentThread.join();
	}
}

void MainWindow::CheckExperimentIdle() {
	const auto& experiment = Config::getInstance().experiment;
	if (!experiment.Enable) {
		return;
	}
	if (autoExportStarted.load()) {
		if (autoExportFinished.load()) {
			if (experimentIdleTimer) {
				experimentIdleTimer->stop();
			}
			StopReceive();
			if (exportThread.joinable()) {
				exportThread.join();
			}
			cout << "[Experiment] Automatic export completed." << endl;
			QCoreApplication::quit();
		}
		return;
	}
	if (!allImagesSubmitted.load() || !workflow || !workflow->IsRegistrationFinalized() ||
		!workflow->IsIdleForExperiment()) return;
	if (!experimentStarted.load() || !imageSubmissionStarted.load() ||
		experimentLogPath.isEmpty()) {
		return;
	}
	const QFileInfo logInfo(experimentLogPath);
	if (!logInfo.exists() || !logInfo.isFile()) {
		return;
	}
	const qint64 idleSeconds = logInfo.lastModified().secsTo(QDateTime::currentDateTime());
	if (idleSeconds < experiment.AutoSaveIdleSeconds) {
		return;
	}
	autoExportStarted.store(true);
	cout << "[Experiment] Log unchanged for "
		<< experiment.AutoSaveIdleSeconds << " seconds. Starting automatic export." << endl;
	ExportProject();
}

void MainWindow::StartReceive() {
	StopExperimentThread();
	stopExperimentRequested.store(false);
	allImagesSubmitted.store(false);
	imageSubmissionStarted.store(false);
	experimentLogPath.clear();
	QSettings settings("HKEY_CURRENT_USER\\Software\\RTPS\\CamFiTransmit", QSettings::NativeFormat);
	settings.setValue("NewImage", "");
	receiver->Start();
	startReceiveAction->setEnabled(false);
	stopReceiverAction->setEnabled(true);

	experimentThread = std::thread(&MainWindow::RunFunc, this);
}

void MainWindow::StopReceive() {
	if (workflow) workflow->SuspendRegistrationRetry();
	stopExperimentRequested.store(true);
	allImagesSubmitted.store(false);
	imageSubmissionStarted.store(false);
	experimentLogPath.clear();
	experimentCondition.notify_all();
	receiver->Stop();
	StopExperimentThread();
	startReceiveAction->setEnabled(true);
	stopReceiverAction->setEnabled(false);

	std::thread tryMergeThread([&]() {
		//reconstructor->TryMergeModels();
		//reconstructor->FinalGlobalBA();
		});
	tryMergeThread.detach();
}

// [Deprecated]
void MainWindow::ProcessImage(const std::string& newImagePath) {
	std::cout << "New image: " << newImagePath << std::endl;
	imageListWidget->AddImage(newImagePath);
}

void MainWindow::ExportProject() {
	const bool sending = imageSubmissionStarted.load() && !allImagesSubmitted.load() && !stopExperimentRequested.load();
	if (workflow && !sending && !allImagesSubmitted.load()) workflow->NotifyInputFinished(0);
	if (workflow && (sending || !workflow->IsRegistrationFinalized() || !workflow->IsIdleForExperiment())) {
		// Keep the event loop running: registration callbacks use BlockingQueuedConnection.
		if (!exportPending) {
			exportPending = true;
			LOG(INFO) << "[Export] Waiting for registration finalization and downstream tasks.";
		}
		if (!exportRetryScheduled) {
			exportRetryScheduled = true;
			QTimer::singleShot(500, this, [this]() {
				exportRetryScheduled = false;
				ExportProject();
			});
		}
		return;
	}
	exportPending = false;
	if (exportThread.joinable()) {
		exportThread.join();
	}
	const std::shared_ptr<const RenderData> denseRenderData =
		modelViewerWidget ? modelViewerWidget->dense_render_data : nullptr;
	exportThread = std::thread([this, denseRenderData]() {
		const std::string basePath = StringReplace(EnsureTrailingSlash(GetParentDir(Database::exportPath)), "\\", "/");
		if (database) {
			cout << "Exporting database..." << endl;
			database->Export(Database::exportPath);
			cout << "Export database completed!" << endl;
		}
		if (reconstructionManager) {
			cout << "Exporting models..." << endl;
			for (int i = 0; i < reconstructionManager->Size(); i++) {
				const std::string modelExportDir = basePath + "models/" + to_string(i + 1);
				CreateDirIfNotExists(modelExportDir + "/bin", true);
				CreateDirIfNotExists(modelExportDir + "/text", true);
				reconstructionManager->Get(i)->WriteBinary(modelExportDir + "/bin");
				reconstructionManager->Get(i)->WriteText(modelExportDir + "/text");
				reconstructionManager->Get(i)->OutputDebugResult(modelExportDir);
			}
			cout << "Export mdoels completed!" << endl;
		}
		logWidget->SaveLogToFile(basePath + "Log.txt");

		MatchMatrixWidget::SaveMatchMatrix(basePath + "match_matrix.bmp", database);
		SaveDenseReconstructionPly(denseRenderData, basePath + "DenseReconstruction");
		if (autoExportStarted.load()) {
			autoExportFinished.store(true);
		}
		});
}

void MainWindow::ExportRunningProject() {
	std::thread exportProjectThread([&]() {
		const std::string basePath = StringReplace(EnsureTrailingSlash(GetParentDir(Database::exportPath)), "\\", "/");
		if (database) {
			cout << "Exporting database..." << endl;
			database->Export(Database::exportPath);
			cout << "Export database completed!" << endl;
		}
		if (reconstructionManager) {
			cout << "Exporting models..." << endl;
			for (int i = 0; i < reconstructionManager->Size(); i++) {
				const std::string modelExportDir = basePath + "models/" + to_string(i + 1);
				CreateDirIfNotExists(modelExportDir + "/bin", true);
				CreateDirIfNotExists(modelExportDir + "/text", true);
				reconstructionManager->Get(i)->WriteBinary(modelExportDir + "/bin");
				reconstructionManager->Get(i)->WriteText(modelExportDir + "/text");
				reconstructionManager->Get(i)->OutputDebugResult(modelExportDir);
			}
			cout << "Export mdoels completed!" << endl;
		}
		logWidget->SaveLogToFile(basePath + "Log.txt");
		});
	exportProjectThread.detach();
}

void MainWindow::ExportModelOnRegisted() {
	auto& config = Config::getInstance();

	if (config.pipeline.AutoExportModel) {
		const size_t numRegImages = reconstructionManager->Get(0)->NumRegImages();
		std::thread exportProjectThread([&]() {
			const std::string basePath = StringReplace(EnsureTrailingSlash(GetParentDir(Database::exportPath)), "\\", "/");
			const std::string modelExportDir = basePath + "models/" + to_string(numRegImages);
			CreateDirIfNotExists(modelExportDir + "/bin", true);
			CreateDirIfNotExists(modelExportDir + "/text", true);
			reconstructionManager->Get(0)->WriteBinary(modelExportDir + "/bin");
			reconstructionManager->Get(0)->WriteText(modelExportDir + "/text");
			reconstructionManager->Get(0)->OutputDebugResult(modelExportDir);
			});
		exportProjectThread.detach();
	}
}

// ================= UI交互相关 =================
void MainWindow::ChangeCurrentModel() {
	/*if (isTrackModelCheckBox->isChecked()) {
		lock_guard<mutex> lock(renderMutex);
		const int lastRegModelID = Reconstructor::lastRegModelID;
		if (lastRegModelID >= 0 && lastRegModelID < reconstructionManager->Size()) {
			reconstructionManagerWidget->Update();
			reconstructionManagerWidget->SelectReconstruction(lastRegModelID);
			renderNowAction->trigger();
		}
	}
	else {
		reconstructionManagerWidget->Update();
		renderNowAction->trigger();
	}*/
}

void MainWindow::ShowMatchMatrix() {
	if (showMatchMatrixWidget) {
		showMatchMatrixWidget->show();
	}
}

void MainWindow::NewProject() {
	projectWidget->show();
}

void MainWindow::ChangeSelectModel() {
	RenderNow();
}

void MainWindow::RenderNow() {
#ifdef DISABLE_MODELVIEWER
	return;
#endif
	try {
		if (reconstructionManager->Size() == 0) {
			reconstructionManagerWidget->SelectReconstruction(ReconstructionManagerWidget::kNewestReconstructionIdx);
			modelViewerWidget->ClearReconstruction();
			return;
		}
		reconstructionManagerWidget->Update();
		size_t reconstruction_idx = reconstructionManagerWidget->SelectedReconstructionIdx();
		if (reconstruction_idx == ReconstructionManagerWidget::kNewestReconstructionIdx) {
			if (reconstructionManager->Size() > 0) {
				reconstruction_idx = reconstructionManager->Size() - 1;
			}
		}
		LOG(INFO) << "[Render]: Wait for rendering...";
		if (isTrackModelCheckBox->isChecked() &&
			!modelViewerWidget->IsDenseViewEnabled()) {
			modelViewerWidget->reconstruction = reconstructionManager->Get(0);
			modelViewerWidget->ReloadReconstruction();
		}
		LOG(INFO) << "[Render]: Render finished.";
	}
	catch (const std::exception& e) {
		std::cout << "RenderNow unknown exception caught" << endl;
		LOG(INFO) << "Exception: " << e.what();
	}
}

void MainWindow::NewProject_SLOT() {
	exportProjectAction->setEnabled(true);

	if (WorkFlow::Export3DGS())
	{
		std::string dbPath = Database::exportPath;
		std::string baseDir = StringReplace(EnsureTrailingSlash(GetParentDir(dbPath)), "\\", "/");
		std::string gsDir = baseDir + "GaussianSplatting/";
		CreateDirIfNotExists(gsDir, true);
	}
}

void MainWindow::addNewImageInWidget(QString p_imagePath)
{
	std::string newImagePath = p_imagePath.toStdString();
	imageListWidget->AddImage(newImagePath);
}

// ================= 菜单状态栏相关 =================
void MainWindow::onTabChanged(int index) {
	if (tabWidget->currentWidget() == modelViewerWidget) {
		statusLabel->setText(modelViewerWidget->status_label);
	}
}

void MainWindow::OnTrackModelToggled(bool checked)
{
	// TODO: 强制渲染一次模型
}

void MainWindow::onProcessNewImage()
{
	modelViewerWidget->StartTimer();
}

//主界面使用一个QLabel显示状态，子标签页发送status信号主界面接收，主界面切换页面改变页面序号，根据当前
//页面序号接收status字符串，更新status显示内容。不会造成内存泄漏。
void MainWindow::UpdateStatus(const QString& status) {
	if (tabWidget->currentWidget() == modelViewerWidget) {
		statusLabel->setText(status);
	}
}

// ================= 流程控制相关 =================
void MainWindow::UpdateSimulationSpeed(quint32 read, quint32 reconstructed)
{
	bufferController.processed(read, reconstructed);
	double nextInputDelay = bufferController.getInputInterval();
	simulationSpeed.store(nextInputDelay);
}


// ================= 槽函数实现 =================
void MainWindow::OnShowMeshChanged()
{
	if (!modelViewerWidget->render_data) {
		return;
	}
	bool p_showMesh = showMeshAction->isChecked();
	bool p_showQuality = showQualityAction->isChecked();
	bool p_showPath = showGuidePathAction->isChecked();
	modelViewerWidget->ReloadFeedback(p_showMesh, p_showQuality, p_showPath);
}

void MainWindow::OnShowQualityChanged()
{
	if (!modelViewerWidget->render_data) {
		return;
	}
	bool p_showMesh = showMeshAction->isChecked();
	bool p_showQuality = showQualityAction->isChecked();
	bool p_showPath = showGuidePathAction->isChecked();
	modelViewerWidget->ReloadFeedback(p_showMesh, p_showQuality, p_showPath);
}

void MainWindow::OnShowGuidePathChanged()
{
	if (!modelViewerWidget->render_data) {
		return;
	}
	bool p_showMesh = showMeshAction->isChecked();
	bool p_showQuality = showQualityAction->isChecked();
	bool p_showPath = showGuidePathAction->isChecked();
	modelViewerWidget->ReloadFeedback(p_showMesh, p_showQuality, p_showPath);
}

void MainWindow::OnFeedback()
{
	modelViewerWidget->render_data = workflow->GetRenderingData();
	if (isTrackModelCheckBox->isChecked() &&
		!modelViewerWidget->IsDenseViewEnabled()) {
		bool p_showMesh = showMeshAction->isChecked();
		bool p_showQuality = showQualityAction->isChecked();
		bool p_showPath = showGuidePathAction->isChecked();
		modelViewerWidget->ReloadFeedback(p_showMesh, p_showQuality, p_showPath);
	}
}

void MainWindow::OnRenderNow(quint32 p_lastRegistedImageID)
{
	SetModelWidgetLastRegistedImageID(p_lastRegistedImageID);
	RenderNow();
}

void MainWindow::OnDenseScaleInitializedForViewer()
{
#ifdef DISABLE_MODELVIEWER
	return;
#endif
	RenderNow();
	if (modelViewerWidget &&
		!modelViewerWidget->IsDenseViewEnabled()) {
		modelViewerWidget->FitViewToReconstruction();
	}
}

void MainWindow::OnDenseViewChanged()
{
	if (!denseViewAction->isChecked()) {
		modelViewerWidget->SetDenseViewEnabled(false);
		return;
	}

	

	

	modelViewerWidget->SetDenseRenderData(workflow->GetDenseRenderingData());
	modelViewerWidget->SetDenseViewEnabled(true);
	modelViewerWidget->ReloadDenseReconstruction(
		showDensePointAction->isChecked(),
		showDenseMeshAction->isChecked());
}

void MainWindow::OnDenseReconstructionUpdated()
{
	modelViewerWidget->SetDenseRenderData(workflow->GetDenseRenderingData());
	if (denseViewAction->isChecked() && modelViewerWidget->IsDenseViewEnabled()) {
		modelViewerWidget->ReloadDenseReconstruction(
			showDensePointAction->isChecked(),
			showDenseMeshAction->isChecked());
	}
}

void MainWindow::OnShowDensePointChanged()
{

	modelViewerWidget->SetDenseRenderData(workflow->GetDenseRenderingData());
	const bool hasDensePoints = modelViewerWidget->dense_render_data &&
		!modelViewerWidget->dense_render_data->dense_points.empty();
	if (showDensePointAction->isChecked() && !hasDensePoints) {
		QMessageBox::information(this, tr("Dense Reconstruction"), tr("Point cloud has not been generated."));
		showDensePointAction->setChecked(false);
		return;
	}

	if (!denseViewAction->isChecked()) {
		denseViewAction->setChecked(true);
	}
	modelViewerWidget->SetDenseViewEnabled(true);
	modelViewerWidget->ReloadDenseReconstruction(
		showDensePointAction->isChecked(),
		showDenseMeshAction->isChecked());
}

void MainWindow::OnShowDenseMeshChanged()
{

	modelViewerWidget->SetDenseRenderData(workflow->GetDenseRenderingData());
	const bool hasDenseMesh = modelViewerWidget->dense_render_data &&
		!modelViewerWidget->dense_render_data->dense_mesh_data.empty();
	if (showDenseMeshAction->isChecked() && !hasDenseMesh) {
		QMessageBox::information(this, tr("Dense Reconstruction"), tr("Mesh has not been generated."));
		showDenseMeshAction->setChecked(false);
		return;
	}

	if (!denseViewAction->isChecked()) {
		denseViewAction->setChecked(true);
	}
	modelViewerWidget->SetDenseViewEnabled(true);
	modelViewerWidget->ReloadDenseReconstruction(
		showDensePointAction->isChecked(),
		showDenseMeshAction->isChecked());
}





// ================== 组件通讯 ===============
void MainWindow::SetModelWidgetLastRegistedImageID(quint32 p_imageID) {
	modelViewerWidget->lastRegistedImageID.store(p_imageID);
}

