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

get_trace() {
    cat \
        | grep "scope: search" \
        | tail -n 1
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
    echo ${2%?}, ${3#?}
}

for f in $* ; do
    if [[ $f == trace.yaml ]] ; then
        TRACE=$(cat $f | get_trace)
        CONFIG=$(cat ${f//trace/config} | get_config)
        merge $f "$TRACE" "$CONFIG"
    elif tar --list -f $f trace.yaml 1>/dev/null 2>/dev/null ; then
        # old format
        ff=trace.yaml
        TRACE=$(tar -xOzf $f $ff | get_trace)
        CONFIG=$(tar -xOzf $f ${ff//trace/config} | get_config)
        merge $f "$TRACE" "$CONFIG"
    else
        # current format
        for ff in $(tar --list -f $f --wildcards */trace.yaml | grep -ve "/.*/") ; do
            TRACE=$(tar -xOzf $f $ff | get_trace)
            CONFIG=$(tar -xOzf $f ${ff//trace/config} | get_config)
            merge $ff "$TRACE" "$CONFIG"
        done
    fi
done
