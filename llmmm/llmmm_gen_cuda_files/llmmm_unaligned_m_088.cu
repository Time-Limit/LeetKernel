#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_88_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<88> mm_instantiator;

public:
  UnalignedM_88_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_88_MMInstantiatorWrapper__;

}  // namespace LLMMM
