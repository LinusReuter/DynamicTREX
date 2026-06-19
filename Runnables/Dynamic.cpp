#include "Commands/Dynamic.h"
#include "../Helpers/Console/CommandLineParser.h"
#include "../Helpers/MultiThreading.h"
#include "../Shell/Shell.h"

using namespace Shell;

int main(int argc, char** argv) {

    CommandLineParser clp(argc, argv);
    pinThreadToCoreId(clp.value<int>("core", 1));
    checkAsserts();
    ::Shell::Shell shell;

    new RAPTORToDynamic(shell);
    new LoadAndApplyDynamicPartition(shell);
    new BuildDynamicQueryData(shell);
    new BuildInitialTransferStore(shell);
    new SimulateDynamicUpdate(shell);
    new SimulateAndCompareTransferUpdate(shell);
    new SimulateAndCompareTransferUpdates(shell);

    shell.run();
    return 0;
}
