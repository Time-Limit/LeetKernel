#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_92_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<92> mm_instantiator;

public:
  UnalignedM_92_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_92_MMInstantiatorWrapper__;

}  // namespace LLMMM
