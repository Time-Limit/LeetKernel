#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_86_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<86> mm_instantiator;

public:
  UnalignedM_86_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_86_MMInstantiatorWrapper__;

}  // namespace LLMMM
