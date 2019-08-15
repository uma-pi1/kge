import visdom
import json
import os
from kge.config import  Config
from os.path import dirname, abspath


"""
This is a small working version of an environment "serarchenv" in which we can search plots for key, value pairs.
"""
#TODO feature requests (or I just did not found them yet):
#TODO 1.) vis.get_windows(env) .. in the fashion of vis.get_window_data(win) but returns all window names in an env
#TODO 2.) callbacks on plot windows are not supported atm (only close)
#TODO 3.) adding meta information to windows, is there something else than doing it by opts

class VisEnvParser():

    # envdic has the following structure:
    # {envname: {windowtitle:{id:winid, config:configdic}}}
    envdic = {}

    @classmethod
    def parse_envs(cls):
        """ Parses visdom environments from disk and keeps meta informations around.

        TODO 1.) Atm I don't see an internal method to browse an env for windows
        TODO 2.) You might want to be able to parse also when you don't have the plots in RAM

        """

        path = dirname(dirname(dirname(abspath(__file__)))) + "/local/visdomenv"
        envs = os.listdir(path)
        envs.remove("view")
        for env in envs:
            with open(path + "/" + env) as jsonstring:  # Use file to refer to the file object
                envdata = json.load(jsonstring)
                env = env.replace(".json", "")
                if "jsons" in envdata.keys():
                    VisEnvParser.envdic[env] = {}
                    for window_title in list(envdata["jsons"].keys()):
                        window_dic = envdata["jsons"][window_title]
                        win_id =  window_dic["id"]
                        if "config" in list(window_dic.keys()):
                            VisEnvParser.envdic[env][window_title] = {"id" :win_id, "config": window_dic["config"]}
                        else:
                            VisEnvParser.envdic[env][window_title] = {window_title: {"id": win_id, "config": "None"}}

    @classmethod
    def find_matching_windows(cls,query,plot):
        """
        Given a query and a plotname, searches the cls.envdic for windows that match.

        query: Assumes the query is e.g. like "model:complex" or "train.optimizer_args.lr:0.1"
        plot: refers to the window title.

        """
        conf = Config(load_default=False)

        #TODO: maybe its better to assign configs to environments instead of windows
        #TODO: you search through all window configs here which are all the same
        #TODO: so you only would have to think about ho to deal with mixed environments


        query = query.split(":")
        query_key = query[0]
        query_val = query[1]

        # holds tuples (envid, windowid)
        result_list = []
        for env in list(cls.envdic.keys()):
            for window_title in list(cls.envdic[env]):
                if window_title != plot:
                    continue
                if "config" in list(cls.envdic[env][window_title].keys()) and cls.envdic[env][window_title]["config"] != "None":
                    conf.options = cls.envdic[env][window_title]['config']
                    val = conf.get(query_key)
                    if str(val) == query_val:
                        result_list.append((env, cls.envdic[env][window_title]["id"]))
        return result_list


#TODO: put function in the visdom handler
def copy_window_to_env(window_id, oldenvname, newenvname):
    """ Copies a window to a new env."""

    visnew = visdom.Visdom(env=newenvname)
    win_data = visnew.get_window_data(window_id, oldenvname)
    js = json.loads(win_data)
    # plotly uses this structure for plots, see e. g. Visdom documentation
    # when adding some meta data in opts this appears as a dict in "layout"
    data = js["content"]["data"][0]
    layout = js["content"]["layout"]

    #TODO make a generic window name
    visnew._send({'data': [data], 'layout': layout})


def run_search():

    # Collect metadata about the envs
    VisEnvParser.parse_envs()

    vis = visdom.Visdom(env="searchenv")
    # Properties window
    properties = [
            {'type': 'text', 'name': 'Enter query', 'value': 'initial'},
            {'type': 'text', 'name': 'Metric/Plot', 'value': 'initial'},
            {'type': 'button', 'name': 'Press to search', 'value': 'initial'}
    ]
    # note this returns just the string id of the window
    properties_window = vis.properties(properties)
    def properties_callback(event):
        if event['event_type'] == 'PropertyUpdate':
            prop_id = event['propertyId']
            value = event['value']
            # update the value in the field with the users input
            properties[prop_id]['value'] = value
            vis.properties(properties, win=properties_window)
            # user clicked search button, start search
            if prop_id == 2:
                windows_to_draw = VisEnvParser.find_matching_windows(properties[0]['value'], properties[1]['value'])
                env = event["eid"]
                for (envname, window_id) in windows_to_draw:
                    copy_window_to_env(window_id, oldenvname=envname, newenvname=env)

    vis.register_event_handler(properties_callback, properties_window)


    input("waiting for callbacks")



if __name__ == '__main__':

    import subprocess
    from threading import Event
    from os.path import dirname, abspath
    import sys,time
    import argparse
    import visdom

    PATH = sys.base_exec_prefix + '/bin/' + 'visdom'
    envpath =  dirname(dirname(dirname(abspath(__file__)))) + "/local/visdomenv"
    process = subprocess.Popen([PATH + " -env_path=" + envpath], shell=True)
    time.sleep(5)

    # DEFAULT_PORT = 8097
    # DEFAULT_HOSTNAME = "http://localhost"
    # parser = argparse.ArgumentParser(description='Demo arguments')
    # parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
    #                     help='port the visdom server is running on.')
    # parser.add_argument('-server', metavar='server', type=str,
    #                     default=DEFAULT_HOSTNAME,
    #                     help='Server address of the target to run the demo on.')
    # parser.add_argument('-base_url', metavar='base_url', type=str,
    #                 default='/',
    #                 help='Base Url.')
    # parser.add_argument('-username', metavar='username', type=str,
    #                 default='',
    #                 help='username.')
    # parser.add_argument('-password', metavar='password', type=str,
    #                 default='',
    #                 help='password.')
    # parser.add_argument('-use_incoming_socket', metavar='use_incoming_socket', type=bool,
    #                 default=True,
    #                 help='use_incoming_socket.')
    # FLAGS = parser.parse_args()

    try:
        # viz = visdom.Visdom(port=FLAGS.port, server=FLAGS.server, base_url=FLAGS.base_url, \
        #         use_incoming_socket=FLAGS.use_incoming_socket)
        run_search()
    except Exception as e:
        print(
            "The visdom experienced an exception while running: {}\n"
            "The demo displays up-to-date functionality with the GitHub "
            "version, which may not yet be pushed to pip. Please upgrade "
            "using `pip install -e .` or `easy_install .`\n"
            "If this does not resolve the problem, please open an issue on "
            "our GitHub.".format(repr(e))
        )
    try:
        Event().wait()
    except:
        process.kill()
