#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_3_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<3> mm_instantiator;

public:
  UnalignedM_3_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_3_MMInstantiatorWrapper__;

}  // namespace LLMMM
