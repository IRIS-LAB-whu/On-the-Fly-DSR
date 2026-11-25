#include "../Base/config.h"
#include "../Base/logging.h"
#include "MainWindow.h"
#include <QApplication>          
#include <tbb/global_control.h>
#include <windows.h>

class SleepGuard : public QObject
{
public:
	SleepGuard(QObject* parent = nullptr) : QObject(parent) {
#ifdef Q_OS_WIN
		SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
#endif
	}

	~SleepGuard() {
#ifdef Q_OS_WIN
		SetThreadExecutionState(ES_CONTINUOUS);
#endif
	}
};

int main(int argc, char* argv[])
{
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

	// ��Դ����
	//tbb::global_control tbb_limit(
	//	tbb::global_control::max_allowed_parallelism,
	//	8
	//);

	Config::getInstance("../settings.ini");
	//Config::getInstance("D:/OntheFlySfM/OntheFlySfM/x64/settings.ini");
	InitializeGlog(argv);

	QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
	QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

	QApplication App(argc, argv);
	SleepGuard guard;

	MainWindow w;
	const auto& experiment = Config::getInstance().experiment;
	if (!experiment.Enable || experiment.ShowUI) {
		w.show();
	}

	return App.exec();
}
