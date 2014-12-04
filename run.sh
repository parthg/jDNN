#!/bin/bash
if [ $# -gt 0 ]; then
  java -classpath web/WEB-INF/classes/:/home/parth/workspace/Nemo/web/WEB-INF/classes/:/home/parth/workspace/cl-deep/bin/:web/WEB-INF/lib/trove-3.0.1.jar:web/WEB-INF/lib/jblas-1.2.3-SNAPSHOT.jar:web/WEB-INF/lib/terrier-3.5-core.jar:web/WEB-INF/lib/antlr.jar:web/WEB-INF/lib/hadoop-0.20.2+228-core.jar:web/WEB-INF/lib/log4j-1.2.15.jar:web/WEB-INF/lib/snowball-20071024.jar:web/WEB-INF/lib/trove-2.0.2.jar:web/WEB-INF/lib/commons-logging-1.1.1.jar $*
else
  echo "Your command line contains no arguments"
fi

