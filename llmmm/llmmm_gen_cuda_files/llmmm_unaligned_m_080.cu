#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_80_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<80> mm_instantiator;

public:
  UnalignedM_80_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_80_MMInstantiatorWrapper__;

}  // namespace LLMMM
