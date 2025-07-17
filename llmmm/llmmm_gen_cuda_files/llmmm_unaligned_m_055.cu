#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_55_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<55> mm_instantiator;

public:
  UnalignedM_55_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_55_MMInstantiatorWrapper__;

}  // namespace LLMMM
