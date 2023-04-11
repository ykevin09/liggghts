//
// Created by Ming Yang on 9/28/22.
//

#include "compute_fabric_atom.h"
#include <cstring>
#include <cmath>
#include "atom.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "comm.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "update.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeFabricAtom::ComputeFabricAtom(LAMMPS * lmp, int &iarg, int narg, char **arg) :
    Compute(lmp, iarg, narg, arg),
    fabric(nullptr)
{
    if (narg < iarg) error->all(FLERR, "Illegal compute fabric/atom command");

    peratom_flag = 1;
    size_peratom_cols = 6;
    timeflag = 1;
    comm_forward = 6;
    comm_reverse = 6;

    nmax = 0;

    // error checks
    if (!atom->sphere_flag)
        error->all(FLERR, "Compute fabric/atom requires atom style sphere");
}

/* ---------------------------------------------------------------------- */

ComputeFabricAtom::~ComputeFabricAtom()
{
    memory->destroy(fabric);
}

/* ---------------------------------------------------------------------- */

void ComputeFabricAtom::init()
{
    if (force->pair == nullptr)
        error->all(FLERR, "Compute fabric/atom requires a pair style be defined");

    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "fabric/atom") == 0) count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute fabric/atom");

    // need an occasional neighbor list

    int irequest = neighbor->request((void *)this);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->gran = 1;
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeFabricAtom::init_list(int id, class NeighList * ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeFabricAtom::compute_peratom()
{
    int i, j, ii, jj, inum, jnum;
    double xtmp, ytmp, ztmp, delx, dely, delz;
    double radi, rsq, radsum, r, rinv;
    double nx, ny, nz;
    int *ilist, *jlist, *numneigh, **firstneigh;

    invoked_peratom = update->ntimestep;

    // grow contact array if necessary

    if (atom->nmax > nmax) {
        memory->destroy(fabric);
        nmax = atom->nmax;
        memory->create(fabric, nmax, 6, "fabric/atom:fabric");
        array_atom = fabric;
    }

    // invoke neighbor list

    neighbor->build_one(list->index);

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // compute
    double **x = atom->x;
    double *radius = atom->radius;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int nall = nlocal + atom->nghost;

    for (i = 0; i < nall; i++)
        for (j = 0; j < 6; j++)
            fabric[i][j] = 0.0;

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {
            xtmp = x[i][0];
            ytmp = x[i][1];
            ztmp = x[i][2];
            radi = radius[i];
            jlist = firstneigh[i];
            jnum = numneigh[i];

            for (jj = 0; jj < jnum; jj++) {
                j = jlist[jj];
                j &= NEIGHMASK;

                delx = xtmp - x[j][0];
                dely = ytmp - x[j][1];
                delz = ztmp - x[j][2];
                rsq = delx*delx + dely*dely + delz*delz;

                r = sqrt(rsq);
                rinv = 1.0 / r;

                nx = delx*rinv;
                ny = dely*rinv;
                nz = delz*rinv;
                radsum = radi + radius[j];
                if (rsq <= radsum*radsum) {
                    fabric[i][0] += nx * nx;
                    fabric[i][1] += ny * ny;
                    fabric[i][2] += nz * nz;
                    fabric[i][3] += nx * ny;
                    fabric[i][4] += nx * nz;
                    fabric[i][5] += ny * nz;

                    fabric[j][0] += nx * nx;
                    fabric[j][1] += ny * ny;
                    fabric[j][2] += nz * nz;
                    fabric[j][3] += nx * ny;
                    fabric[j][4] += nx * nz;
                    fabric[j][5] += ny * nz;
                }
            }
        }
    }

    // communicate ghost atom counts between neighbor procs if necessary

    if (force->newton_pair) comm->reverse_comm_compute(this);
}

/* ---------------------------------------------------------------------- */

int ComputeFabricAtom::pack_reverse_comm(int n, int first, double *buf)
{
    int i, m, last;
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
        buf[m++] = fabric[i][0];
        buf[m++] = fabric[i][1];
        buf[m++] = fabric[i][2];
        buf[m++] = fabric[i][3];
        buf[m++] = fabric[i][4];
        buf[m++] = fabric[i][5];
    }
    return 6;
}

/* ---------------------------------------------------------------------- */

void ComputeFabricAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
    int i, j, m;

    m = 0;
    for (i = 0; i < n; i++) {
        j = list[i];
        fabric[j][0] += buf[m++];
        fabric[j][1] += buf[m++];
        fabric[j][2] += buf[m++];
        fabric[j][3] += buf[m++];
        fabric[j][4] += buf[m++];
        fabric[j][5] += buf[m++];
    }
}

/* ---------------------------------------------------------------------- */

double ComputeFabricAtom::memory_usage()
{
    double bytes = (double)nmax*6 * sizeof(double);
    return bytes;
}

