#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_2_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<2> mm_instantiator;

public:
  UnalignedM_2_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_2_MMInstantiatorWrapper__;

}  // namespace LLMMM
