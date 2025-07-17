#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_61_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<61> mm_instantiator;

public:
  UnalignedM_61_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_61_MMInstantiatorWrapper__;

}  // namespace LLMMM
