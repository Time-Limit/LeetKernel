#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_65_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<65> mm_instantiator;

public:
  UnalignedM_65_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_65_MMInstantiatorWrapper__;

}  // namespace LLMMM
