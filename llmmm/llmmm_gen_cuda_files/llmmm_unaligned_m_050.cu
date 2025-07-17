#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_50_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<50> mm_instantiator;

public:
  UnalignedM_50_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_50_MMInstantiatorWrapper__;

}  // namespace LLMMM
