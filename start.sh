#!/bin/bash

if [ "$TASK" = "1" ]; then
    python src/task_1.py; 
elif [ "$TASK" = "4" ]; then 
    python -u src/task_4.py; 
else
    echo "Set the environment variable \$TASK to either '1' or '4'"
fi