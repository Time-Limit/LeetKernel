#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_72_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<72> mm_instantiator;

public:
  UnalignedM_72_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_72_MMInstantiatorWrapper__;

}  // namespace LLMMM
