# ================================================================================================================
# Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
# However, this is "infinitely" cheaper than all the other computations we make around here ;-)
# Ugly, but necessary...
# From:
#   https://stackoverflow.com/questions/44131691/how-to-clear-cache-or-force-recompilation-in-numba
# ================================================================================================================
import os


def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "/../../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)
