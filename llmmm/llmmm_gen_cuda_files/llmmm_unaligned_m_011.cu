#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_11_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<11> mm_instantiator;

public:
  UnalignedM_11_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_11_MMInstantiatorWrapper__;

}  // namespace LLMMM
