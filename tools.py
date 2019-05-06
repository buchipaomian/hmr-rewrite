from os.path import join
from glob import glob


def get_all_files(data_dir,file_format):
    # Dataset with different name path
    data = ""
    if file_format == "img":
        data = join(data_dir,'*.img')
    elif file_format == "csv":
        data = join(data_dir,'*.csv')
    
    # data_dirs = [
    #     join(dataset_dir, dataset, '%s_*.tfrecord' % split)
    #     for dataset in datasets if dataset not in diff_name
    # ]
    # if 'h36m' in datasets:
    #     data_dirs.append(
    #         join(dataset_dir, 'tf_records_human36m_wjoints', split,
    #              '*.tfrecord'))
    # if 'mpi_inf_3dhp' in datasets:
    #     data_dirs.append(
    #         join(dataset_dir, 'mpi_inf_3dhp', split, '*.tfrecord'))

    all_files = []
    all_files += sorted(glob(data))

    return all_files