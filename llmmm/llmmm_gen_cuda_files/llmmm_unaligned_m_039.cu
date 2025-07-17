#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_39_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<39> mm_instantiator;

public:
  UnalignedM_39_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_39_MMInstantiatorWrapper__;

}  // namespace LLMMM
