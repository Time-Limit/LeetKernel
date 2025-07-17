#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_91_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<91> mm_instantiator;

public:
  UnalignedM_91_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_91_MMInstantiatorWrapper__;

}  // namespace LLMMM
