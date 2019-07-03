#!/bin/bash
#
# Extracts the trace entry summarizing the search results of one or more search
# jobs. Adds all configuration parameters that have atomic values. Each argument
# is either:
#
# 1. A trace file of a search job.
# 2. A .tgz file holding the result of a serach job (at least: trace.yaml, config.yaml)
# 3. A .tgz file with multiple subdirectories, each holding the result of a search job.
set -e

get_jobid() {
    cat | tail -1 | sed -e "s/^.*job_id: \([^,]*\).*$/\1/"
}

get_trace() {
    cat \
        | grep "$1"
}

get_config() {
    cat \
        | python -c "import yaml, sys; from kge import Config; options=yaml.load(sys.stdin, Loader=yaml.SafeLoader); print(yaml.dump({k:v for k,v in Config.flatten(options).items() if type(v) in [ str, int, float ]}, width=float('inf'), default_flow_style=True))"
}

merge() {
    if [ "" == "$2" ] ; then
        echo "Skipping $1 (no trace entry found)" 1>&2
        return
    fi
    if [ "" == "$3" ] ; then
        echo "Skipping $1 (no config found)" 1>&2
        return
    fi
    IFS=$'\n'
    for line in $2; do
        A=${line%?}
        B=${3#?}
        B=${B%?}
        echo "$A, $B, archive: $4, archive_folder: $5}"
    done
}


for f in $* ; do
    if [[ $f == trace.yaml ]] ; then
        JOBID=$(cat $f | get_jobid)
        TRACE=$(cat $f | get_trace $JOBID)
        CONFIG=$(cat ${f//trace/config} | get_config)
        merge $f "$TRACE" "$CONFIG" "$(basename $(pwd))" "."
    elif tar --list -f $f trace.yaml 1>/dev/null 2>/dev/null ; then
        # old format
        ff=trace.yaml
        JOBID=$(tar -xOzf $f $ff | get_jobid)
        TRACE=$(tar -xOzf $f $ff | get_trace $JOBID)
        CONFIG=$(tar -xOzf $f ${ff//trace/config} | get_config)
        merge $f "$TRACE" "$CONFIG" "$f" "."
    else
        # current format
        for ff in $(tar --list -f $f --wildcards "*/trace.yaml" | grep -ve "/.*/") ; do
            JOBID=$(tar -xOzf $f $ff | get_jobid)
            TRACE=$(tar -xOzf $f $ff | get_trace $JOBID)
            CONFIG=$(tar -xOzf $f ${ff//trace/config} | get_config)
            merge $ff "$TRACE" "$CONFIG" "$f" "$(dirname $ff)"
        done
    fi
done
