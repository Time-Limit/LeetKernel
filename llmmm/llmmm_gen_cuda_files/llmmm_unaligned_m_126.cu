#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_126_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<126> mm_instantiator;

public:
  UnalignedM_126_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_126_MMInstantiatorWrapper__;

}  // namespace LLMMM
