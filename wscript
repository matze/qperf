SOURCES = [
    'qperf.c'
]

OPENCL_INC_PATHS = [
    '/usr/local/cuda/include',
    '/opt/cuda/include'
]


def guess_cl_include_path():
    import os

    try:
        OPENCL_INC_PATHS.append(os.environ['CUDA_INC_PATH'])
    except:
        pass

    return filter(lambda d: os.path.exists(d), OPENCL_INC_PATHS)


def options(opt):
    opt.load('compiler_c')


def configure(conf):
    conf.load('compiler_c')
    conf.env.append_unique('CFLAGS', ['-g', '-ggdb', '-std=c99', '-O3', '-Wall', '-Werror' ])

    # Check OpenCL include paths ...
    conf.start_msg('Checking for OpenCL include path')
    incs = guess_cl_include_path()

    if incs:
        conf.env.OPENCL_INC_PATH = incs[0]
        conf.end_msg('yes')
    else:
        conf.fatal('OpenCL include path not found')

    # ... and library
    conf.check_cc(lib='OpenCL', uselib_store='CL')

    # Check libraries using pkg-config
    conf.check_cfg(package='glib-2.0', args='--cflags --libs', uselib_store='GLIB2')


def build(bld):
    srcs = ' '.join(SOURCES)
    incs = ' '.join(OPENCL_INC_PATHS)
    bld.program(source=srcs, target='qperf', use=['GLIB2', 'CL'], includes=bld.env.OPENCL_INC_PATH)
