#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_48_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<48> mm_instantiator;

public:
  UnalignedM_48_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_48_MMInstantiatorWrapper__;

}  // namespace LLMMM
