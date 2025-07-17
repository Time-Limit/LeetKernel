#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_118_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<118> mm_instantiator;

public:
  UnalignedM_118_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_118_MMInstantiatorWrapper__;

}  // namespace LLMMM
