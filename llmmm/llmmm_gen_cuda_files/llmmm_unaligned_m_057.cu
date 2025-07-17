#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_57_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<57> mm_instantiator;

public:
  UnalignedM_57_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_57_MMInstantiatorWrapper__;

}  // namespace LLMMM
