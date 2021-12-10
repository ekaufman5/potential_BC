"""
Plot meridional slices

Usage:
    plot_meridian.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames_meridian]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt

from dedalus.extras import plot_tools

def build_mer_coord_vertices(theta, r):
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    r = r.ravel()
    r_mid = (r[:-1] + r[1:]) / 2
    r_vert = np.concatenate([[0], r_mid, [1]])
    return np.meshgrid(theta_vert, r_vert, indexing='ij')

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    tasks = ['B_mer(phi=0)', 'B_mer(phi=pi)', 'u_mer(phi=0)', 'u_mer(phi=pi)']
    cmap = plt.cm.viridis
    savename_func = lambda write: 'meridian_{:06}.png'.format(write)
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
#    names = [r'$\rho(\phi=0,\, \pi)$', r'$B_r(\phi=0,\,\pi)$', r'$u_r(\phi=0,\,\pi)$']
    names = [r'$B_r(\phi=0,\,\pi)$', r'$u_r(\phi=0,\,\pi)$']

    # Create figure
    nrows, ncols = 1, 2
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0.1, 0.1, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)
    scale = 2.5
    dpi = 200

    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    paxes = []
    cbaxes = []
    for n in range(ncols*nrows):
        i, j = divmod(n, ncols)
        paxes.append(mfig.add_axes(i, j, [0.03, 0, 0.94, 0.94]))
        cbaxes.append(mfig.add_axes(i, j, [0.03, 0.95, 0.94, 0.05]))

    # Plot writes
    with h5py.File(filename, mode='r') as file:

        title_text = title_func(file['scales/sim_time'][start])
        title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
        title = fig.suptitle(title_text, x=0.48, y=title_height, ha='left')

        dsets = []
        xs = []
        ys = []
        for i, task in enumerate(tasks):
            dsets.append(file['tasks'][task])
            theta = dsets[i].dims[2][0][:].ravel()
            r = dsets[i].dims[3][0][:].ravel()
            theta_vert, r_vert = build_mer_coord_vertices(theta, r)
            xs.append(np.sin(theta_vert) * r_vert * (1 - 2*(i % 2)))
            ys.append(np.cos(theta_vert) * r_vert)
        pcms = []
        cbs = []
        for index in range(start, start+count):
            print(index)
            for i_plot in range(len(paxes)):
                if i_plot == 0:
                    data_slices = (index, 2, 0, slice(None), slice(None)) #change 2nd value to change what gets plotted
                else:
                    data_slices = (index, 2, 0, slice(None), slice(None))
                data_min = min(dsets[2*i_plot][data_slices].min(), dsets[2*i_plot+1][data_slices].min())
                data_max = max(dsets[2*i_plot][data_slices].max(), dsets[2*i_plot+1][data_slices].max())
                for i in range(2):
                    i_dset = 2*i_plot+i
                    data = dsets[i_dset][data_slices]
                    if index == start:
                        pcms.append( paxes[i_plot].pcolormesh(xs[i_dset], ys[i_dset], data, cmap=cmap, vmin=data_min, vmax=data_max) )
                        if i == 0:
                            cbs.append( plt.colorbar(pcms[i_dset], cax=cbaxes[i_plot], orientation='horizontal',
                                                     ticks=ticker.MaxNLocator(nbins=5)) )
                            cbs[-1].outline.set_visible(False)
                            cbaxes[i_plot].xaxis.set_ticks_position('top')
                            cbaxes[i_plot].xaxis.set_label_position('top')
                            cbaxes[i_plot].set_xlabel(names[i_plot])
                            paxes[i_plot].set_ylabel(r'$z$')
                            paxes[i_plot].set_xlabel(r'$x$')
                    else:
                        pcms[i_dset].set_array(np.ravel(data))
                        pcms[i_dset].set_clim(data_min, data_max)
                        if i == 0:
                            cbaxes[i_plot].xaxis.set_ticks_position('top')
                            cbaxes[i_plot].xaxis.set_label_position('top')
                            cbaxes[i_plot].set_xlabel(names[i_plot])

            title_text = title_func(file['scales/sim_time'][index])
            title.set_text(title_text)

            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

