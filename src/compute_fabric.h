//
// Created by Ming Yang on 9/28/22.
//

#ifdef COMPUTE_CLASS

ComputeStyle(fabric, ComputeFabric);

#else

#ifndef LMP_COMPUTE_FABRIC_H
#define LMP_COMPUTE_FABRIC_H

#include "compute.h"
#include "compute_pair_gran_local.h"

namespace LAMMPS_NS {

    class ComputeFabric : public Compute {
    public:
        ComputeFabric(class LAMMPS *, int &, int, char **);
        ~ComputeFabric() override;
        void init() override;
        // void init_list(int, class NeighList *) override;
        void compute_vector() override;

    private:
        int ntensors, pstyle, cutstyle;
        long int nc;
        int *tensor_style;
        int **type_filter;
        // class NeighList *list;

        int cn_flag, br_flag, fn_flag, ft_flag;

        class ComputePairGranLocal *cpgl_;
        int offset_x1, offset_fn, offset_ft;
    };
}

#endif //LMP_COMPUTE_FABRIC_H
#endif
