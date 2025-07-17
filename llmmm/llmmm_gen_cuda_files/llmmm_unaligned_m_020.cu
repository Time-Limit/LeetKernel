#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_20_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<20> mm_instantiator;

public:
  UnalignedM_20_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_20_MMInstantiatorWrapper__;

}  // namespace LLMMM
