//
// Created by Ming Yang on 9/28/22.
//
#ifdef COMPUTE_CLASS

ComputeStyle(fabric/atom, ComputeFabricAtom)

#else

#ifndef LMP_COMPUTE_FABRIC_ATOM_H
#define LMP_COMPUTE_FABRIC_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

    class ComputeFabricAtom : public Compute {
    public:
        ComputeFabricAtom(class LAMMPS *, int &, int, char **);
        ~ComputeFabricAtom();
        void init() override;
        void init_list(int, class NeighList *) override;
        void compute_peratom() override;
        int pack_reverse_comm(int, int, double *) override;
        void unpack_reverse_comm(int, int *, double *) override;
        double memory_usage() override;

    private:
        int nmax;
        class NeighList *list;
        double **fabric;

    };
}

#endif //LMP_COMPUTE_FABRIC_ATOM_H
#endif
