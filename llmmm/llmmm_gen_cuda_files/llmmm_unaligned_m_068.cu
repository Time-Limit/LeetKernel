#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_68_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<68> mm_instantiator;

public:
  UnalignedM_68_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_68_MMInstantiatorWrapper__;

}  // namespace LLMMM
