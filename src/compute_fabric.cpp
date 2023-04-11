//
// Created by Ming Yang on 9/28/22.
//

#include "compute_fabric.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "compute_pair_gran_local.h"
#include "pair_gran.h"

#include <cstring>
#include <cmath>

using namespace LAMMPS_NS;

enum {OTHER, GRANULAR };
enum { TYPE, RADIUS };
enum { CN, BR, FN, FT };

/* ---------------------------------------------------------------------- */

ComputeFabric::ComputeFabric(LAMMPS * lmp, int &iarg, int narg, char **arg) :
    Compute(lmp, iarg, narg, arg), pstyle(0), nc(0)
{
    if (narg < iarg) error->all(FLERR, "Illegal compute fabric command");

    if (strcmp(arg[iarg], "type") == 0)
        cutstyle = TYPE;
    else if (strcmp(arg[iarg], "radius") == 0)
        cutstyle = RADIUS;
    else
        error->all(FLERR, "Illegal compute fabric command");

    if (cutstyle == RADIUS && !atom->radius_flag)
        error->all(FLERR, "Compute fabric radius style requires atom attribute radius");

    // If optional arguments included, this will be oversized
    ntensors = narg - 4;
    tensor_style = new int[ntensors];

    cn_flag = 0;
    br_flag = 0;
    fn_flag = 0;
    ft_flag = 0;
    type_filter = nullptr;

    ntensors = 0;
    iarg += 1;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "contact") == 0) {
            cn_flag = 1;
            tensor_style[ntensors++] = CN;
        }
        else if (strcmp(arg[iarg], "branch") == 0) {
            br_flag = 1;
            tensor_style[ntensors++] = BR;
        }
        else if (strcmp(arg[iarg], "force/normal") == 0) {
            fn_flag = 1;
            tensor_style[ntensors++] = FN;
        }
        else if (strcmp(arg[iarg], "force/tangential") == 0) {
            ft_flag = 1;
            tensor_style[ntensors++] = FT;
        }
        else {
            error->all(FLERR, "Illegal compute fabric command");
        }
        iarg++;
    }

    vector_flag = 1;
    size_vector = ntensors * 6;
    extvector = 0;

    scalar_flag = 1;
    extscalar = 1;

    vector = new double [size_vector];

    // list = nullptr;

    if ((fn_flag || ft_flag) && ((PairGran *)(force->pair))->cpl() == nullptr)
        error->all(FLERR, "Compute fabric requires compute pair/gran/local in advance");

    cpgl_ = ((PairGran *)(force->pair))->cpl();
    if (cpgl_->offset_x1() == -1)
        error->all(FLERR, "Compute fabric requires compute pair/gran/local to output pos");
    if (cpgl_->offset_fn() == -1)
        error->all(FLERR, "Compute fabric requires compute pair/gran/local to output force/normal");
    if (cpgl_->offset_ft() == -1)
        error->all(FLERR, "Compute fabric requires compute pair/gran/local to output force/tangential");

    offset_x1 = cpgl_->offset_x1();
    offset_fn = cpgl_->offset_fn();
    offset_ft = cpgl_->offset_ft();
}

/* ---------------------------------------------------------------------- */

ComputeFabric::~ComputeFabric()
{
    delete [] vector;
    delete [] tensor_style;
    memory->destroy(type_filter);
}

/* ---------------------------------------------------------------------- */

void ComputeFabric::init()
{
    if (force->pair == nullptr) error->all(FLERR, "No pair style is defined for compute fabric");
//    if (force->pair->single_enable == 0 && (fn_flag || ft_flag))
    if (((PairGran *)(force->pair))->cplenable() == 0 && (fn_flag || ft_flag))
        error->all(FLERR, "Pair style does not support compute fabric normal or tangential force");

    // Find if granular or gran
    pstyle = OTHER;
    if (force->pair_match("granular", 0) || force->pair_match("gran", 0)) pstyle = GRANULAR;

    if (pstyle != GRANULAR && ft_flag)
        error->all(FLERR, "Pair style does not calculate tangential forces for compute fabric");

    // need an occasional half neighbor list
    // set size to same value as request made by force->pair
    // this should enable it to always be a copy list (e.g. for granular pstyle)
    int irequest = neighbor->request((void *)this);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->gran = 1;
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

//void ComputeFabric::init_list(int /*id*/, class NeighList *ptr)
//{
//    list = ptr;
//}

/* ---------------------------------------------------------------------- */

void ComputeFabric::compute_vector()
{
    invoked_vector = update->ntimestep;

    int i, j, nc_local;
    double delx, dely, delz, r, rinv;

    double nx, ny, nz, fnx, fny, fnz, ftx, fty, ftz;
    double ncinv, denom, fn, fninv, prefactor;
    double br_tensor[6], ft_tensor[6], fn_tensor[6];
    double trace_phi, trace_phid, trace_phin;
    double phi_ij[6] = {0.0};
    double Ac_ij[6] = {0.0};
    double phid_ij[6] = {0.0};
    double phin_ij[6] = {0.0};
    double phit_ij[6] = {0.0};
    double temp_dbl[6];

    if (cpgl_->invoked_local != update->ntimestep)
        cpgl_->compute_local();

    //No. of contacts
    nc_local = cpgl_->get_ncount();
    //Count total contacts across processors
    MPI_Allreduce(&nc_local, &nc, 1, MPI_INT, MPI_SUM, world);

    double **data;
    data = static_cast<double **>(cpgl_->get_data());

    if (nc == 0) {
        for (i = 0; i < size_vector; i++) vector[i] = 0.0;
        return;
    }
    ncinv = 1.0 / static_cast<double>(nc);

    for (size_t i_c = 0; i_c < nc_local; i_c++) {
        /*delx = data[i_c][offset_x1]-data[i_c][offset_x1+3];
        dely = data[i_c][offset_x1+1]-data[i_c][offset_x1+4];
        delz = data[i_c][offset_x1+2]-data[i_c][offset_x1+5];
        r = sqrt(delx*delx + dely*dely + delz*delz);
        rinv = 1./r;

        nx = delx * rinv;
        ny = dely * rinv;
        nz = delz * rinv;*/

        fnx = data[i_c][offset_fn];
        fny = data[i_c][offset_fn+1];
        fnz = data[i_c][offset_fn+2];
        fn = sqrt(fnx*fnx+fny*fny+fnz*fnz);
        if (fn > 0) {
            fninv = 1./fn;
            nx = fnx*fninv;
            ny = fny*fninv;
            nz = fnz*fninv;

            phi_ij[0] += nx * nx;
            phi_ij[1] += ny * ny;
            phi_ij[2] += nz * nz;
            phi_ij[3] += nx * ny;
            phi_ij[4] += nx * nz;
            phi_ij[5] += ny * nz;
        }
    }

    // Sum phi across processors
    MPI_Allreduce(phi_ij, temp_dbl, 6, MPI_DOUBLE, MPI_SUM, world);
    for (i = 0; i < 6; i++) phi_ij[i] = temp_dbl[i] * ncinv;

    trace_phi = (1.0 / 3.0) * (phi_ij[0] + phi_ij[1] + phi_ij[2]);

    Ac_ij[0] = (15.0 / 2.0) * (phi_ij[0] - trace_phi);
    Ac_ij[1] = (15.0 / 2.0) * (phi_ij[1] - trace_phi);
    Ac_ij[2] = (15.0 / 2.0) * (phi_ij[2] - trace_phi);
    Ac_ij[3] = (15.0 / 2.0) * (phi_ij[3]);
    Ac_ij[4] = (15.0 / 2.0) * (phi_ij[4]);
    Ac_ij[5] = (15.0 / 2.0) * (phi_ij[5]);

    // If needed, loop through and calculate other fabric tensors
    if (br_flag || fn_flag || ft_flag) {
        for (size_t i_c = 0; i_c < nc_local; i_c++) {
            delx = data[i_c][offset_x1]-data[i_c][offset_x1+3];
            dely = data[i_c][offset_x1+1]-data[i_c][offset_x1+4];
            delz = data[i_c][offset_x1+2]-data[i_c][offset_x1+5];
            r = sqrt(delx*delx + dely*dely + delz*delz);
            rinv = 1./r;

            nx = delx * rinv;
            ny = dely * rinv;
            nz = delz * rinv;

            denom = 1.0 + Ac_ij[0]*nx*nx + Ac_ij[1]*ny*ny + Ac_ij[2]*nz*nz;
            denom += 2.0 * Ac_ij[3]*nx*ny + 2.0*Ac_ij[4]*nx*nz + 2.0*Ac_ij[5]*ny*nz;
            prefactor = ncinv / denom;

            if (br_flag) {
                phid_ij[0] += prefactor * nx * nx * r;
                phid_ij[1] += prefactor * ny * ny * r;
                phid_ij[2] += prefactor * nz * nz * r;
                phid_ij[3] += prefactor * nx * ny * r;
                phid_ij[4] += prefactor * nx * nz * r;
                phid_ij[5] += prefactor * ny * nz * r;
            }

            if (fn_flag || ft_flag) {
                fnx = data[i_c][offset_fn];
                fny = data[i_c][offset_fn+1];
                fnz = data[i_c][offset_fn+2];
                fn = sqrt(fnx*fnx + fny*fny + fnz*fnz);

                phin_ij[0] += prefactor * nx * nx * fn;
                phin_ij[1] += prefactor * ny * ny * fn;
                phin_ij[2] += prefactor * nz * nz * fn;
                phin_ij[3] += prefactor * nx * ny * fn;
                phin_ij[4] += prefactor * nx * nz * fn;
                phin_ij[5] += prefactor * ny * nz * fn;

                if (ft_flag) {
                    ftx = data[i_c][offset_ft];
                    fty = data[i_c][offset_ft+1];
                    ftz = data[i_c][offset_ft+2];

                    phit_ij[0] += prefactor * ftx * nx;
                    phit_ij[1] += prefactor * fty * ny;
                    phit_ij[2] += prefactor * ftz * nz;
                    phit_ij[3] += prefactor * ftx * ny;
                    phit_ij[4] += prefactor * ftx * nz;
                    phit_ij[5] += prefactor * fty * nz;
                }
            }
        }
    }

    // output results

    if (cn_flag) {
        for (i = 0; i < ntensors; i++) {
            if (tensor_style[i] == CN) {
                for (j = 0; j < 6; j++) vector[6 * i + j] = Ac_ij[j];
            }
        }
    }

    if (br_flag) {
        MPI_Allreduce(phid_ij, temp_dbl, 6, MPI_DOUBLE, MPI_SUM, world);
        for (i = 0; i < 6; i++) phid_ij[i] = temp_dbl[i];

        trace_phid = (1.0 / 3.0) * (phid_ij[0] + phid_ij[1] + phid_ij[2]);

        br_tensor[0] = (15.0 / (6.0 * trace_phid)) * (phid_ij[0] - trace_phid);
        br_tensor[1] = (15.0 / (6.0 * trace_phid)) * (phid_ij[1] - trace_phid);
        br_tensor[2] = (15.0 / (6.0 * trace_phid)) * (phid_ij[2] - trace_phid);
        br_tensor[3] = (15.0 / (6.0 * trace_phid)) * (phid_ij[3]);
        br_tensor[4] = (15.0 / (6.0 * trace_phid)) * (phid_ij[4]);
        br_tensor[5] = (15.0 / (6.0 * trace_phid)) * (phid_ij[5]);

        for (i = 0; i < ntensors; i++) {
            if (tensor_style[i] == BR) {
                for (j = 0; j < 6; j++) vector[6 * i + j] = br_tensor[j];
            }
        }
    }

    if (fn_flag || ft_flag) {
        MPI_Allreduce(phin_ij, temp_dbl, 6, MPI_DOUBLE, MPI_SUM, world);
        for (i = 0; i < 6; i++) phin_ij[i] = temp_dbl[i];

        trace_phin = (1.0 / 3.0) * (phin_ij[0] + phin_ij[1] + phin_ij[2]);
    }

    if (fn_flag) {
        fn_tensor[0] = (15.0 / (6.0 * trace_phin)) * (phin_ij[0] - trace_phin);
        fn_tensor[1] = (15.0 / (6.0 * trace_phin)) * (phin_ij[1] - trace_phin);
        fn_tensor[2] = (15.0 / (6.0 * trace_phin)) * (phin_ij[2] - trace_phin);
        fn_tensor[3] = (15.0 / (6.0 * trace_phin)) * (phin_ij[3]);
        fn_tensor[4] = (15.0 / (6.0 * trace_phin)) * (phin_ij[4]);
        fn_tensor[5] = (15.0 / (6.0 * trace_phin)) * (phin_ij[5]);

        for (i = 0; i < ntensors; i++) {
            if (tensor_style[i] == FN) {
                for (j = 0; j < 6; j++) vector[6 * i + j] = fn_tensor[j];
            }
        }
    }

    if (ft_flag) {
        MPI_Allreduce(phit_ij, temp_dbl, 6, MPI_DOUBLE, MPI_SUM, world);
        for (i = 0; i < 6; i++) phit_ij[i] = temp_dbl[i];

        ft_tensor[0] = (15.0 / (9.0 * trace_phin)) * phit_ij[0];
        ft_tensor[1] = (15.0 / (9.0 * trace_phin)) * phit_ij[1];
        ft_tensor[2] = (15.0 / (9.0 * trace_phin)) * phit_ij[2];
        ft_tensor[3] = (15.0 / (9.0 * trace_phin)) * phit_ij[3];
        ft_tensor[4] = (15.0 / (9.0 * trace_phin)) * phit_ij[4];
        ft_tensor[5] = (15.0 / (9.0 * trace_phin)) * phit_ij[5];

        for (i = 0; i < ntensors; i++) {
            if (tensor_style[i] == FT) {
                for (j = 0; j < 6; j++) vector[6 * i + j] = ft_tensor[j];
            }
        }
    }
}
/* ---------------------------------------------------------------------- */

