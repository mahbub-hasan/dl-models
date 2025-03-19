from project_config import ProjectConfiguration
from load_file import LoadFile

if __name__ == '__main__':
    config = ProjectConfiguration()
    load_file = LoadFile(config=config,
                         csv_file_path='/Users/mahbubhasan/Documents/Research_Self/archive/HAM10000_metadata.csv',
                         image_file_path='/Users/mahbubhasan/Documents/Research_Self/archive/Skin Cancer/Skin Cancer')

    load_file.getDataset()

    # edge_blob = 32
    # step_size = 32
    # row = 128
    # col = 128
    #
    # t = 0
    # for i in range(edge_blob, (row - 2 * edge_blob + 2), step_size):
    #     for j in range(edge_blob, (col - 2 * edge_blob + 2), step_size):

                

