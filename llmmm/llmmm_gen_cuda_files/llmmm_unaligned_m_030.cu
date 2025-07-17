#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_30_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<30> mm_instantiator;

public:
  UnalignedM_30_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_30_MMInstantiatorWrapper__;

}  // namespace LLMMM
