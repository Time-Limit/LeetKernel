#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_114_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<114> mm_instantiator;

public:
  UnalignedM_114_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_114_MMInstantiatorWrapper__;

}  // namespace LLMMM
