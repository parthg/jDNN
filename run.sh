#!/bin/bash
if [ $# -gt 0 ]; then
  java -Xmx28G -Duse_cuda="true" -ea -classpath web/WEB-INF/classes/:web/WEB-INF/lib/Nemo-20150613.jar:web/WEB-INF/lib/cl-deep-20150204.jar:web/WEB-INF/lib/trove-3.0.1.jar:web/WEB-INF/lib/jblas-1.2.3-SNAPSHOT.jar:web/WEB-INF/lib/terrier-3.5-core.jar:web/WEB-INF/lib/antlr.jar:web/WEB-INF/lib/hadoop-0.20.2+228-core.jar:web/WEB-INF/lib/log4j-1.2.15.jar:web/WEB-INF/lib/snowball-20071024.jar:web/WEB-INF/lib/trove-2.0.2.jar:web/WEB-INF/lib/commons-logging-1.1.1.jar:web/WEB-INF/lib/mallet.jar:web/WEB-INF/lib/opennlp-tools-1.5.0.jar:web/WEB-INF/lib/maxent-3.0.0.jar:/home/parth/workspace/JCuda-All-0.5.5-bin-linux-x86_64/jcuda-0.5.5.jar:/home/parth/workspace/JCuda-All-0.5.5-bin-linux-x86_64/jcublas-0.5.5.jar $*
else
  echo "Your command line contains no arguments"
fi

